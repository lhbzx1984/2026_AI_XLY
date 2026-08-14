"""
舌体分割模块
============
采用"多色相候选 + 位置先验 + 形状验证 + 边缘精修"的多阶段分割方法。

主要改进：
1. 多色相检测：同时支持红色、紫色、粉色、暗红等多种舌色（含淡白舌、青紫舌）
2. 位置先验：种子点搜索集中在图像中央偏下区域，避开 UI 与背景
3. 形状评分：对每个候选区域计算椭圆度、长宽比、面积比等形状得分
4. UI 排除：精确排除橙色按钮、白色 UI 文字、图标等
5. 边缘精修：使用 Canny 边缘 + GrabCut 双重精修边界

参考论文：
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts
"""

import cv2
import numpy as np
from PIL import Image

from .utils import load_image, resize_image


# ============================================================
# 边界平滑与轮廓优化辅助函数
# ============================================================

def _smooth_mask(mask: np.ndarray, blur_ksize: int = 7) -> np.ndarray:
    """
    对二值掩膜进行高斯模糊 + 重新二值化，获得平滑的边界。

    参数:
        mask: 输入二值掩膜（0/255）
        blur_ksize: 高斯模糊核大小（奇数）

    返回:
        平滑后的二值掩膜
    """
    if mask.sum() == 0:
        return mask
    # 高斯模糊使边界像素值渐变
    blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    # 重新二值化，渐变区中间值作为阈值
    _, smoothed = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    # 形态学闭运算填充残留小洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
    return smoothed


def _laplacian_smooth_contour(contour: np.ndarray, iterations: int = 5,
                               alpha: float = 0.25) -> np.ndarray:
    """
    对轮廓点进行拉普拉斯平滑（每个点向邻居中点靠拢），消除锯齿。

    参数:
        contour: OpenCV 轮廓，形状 (N, 1, 2)
        iterations: 平滑迭代次数
        alpha: 平滑强度（0-1，越大越平滑）

    返回:
        平滑后的轮廓 (N, 1, 2) int32
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 4:
        return contour

    for _ in range(iterations):
        new_pts = pts.copy()
        for i in range(n):
            prev = pts[(i - 1) % n]
            nxt = pts[(i + 1) % n]
            # 拉普拉斯：当前点向前后两点的中点移动
            mid = (prev + nxt) * 0.5
            new_pts[i] = pts[i] * (1 - alpha) + mid * alpha
        pts = new_pts

    return pts.reshape(-1, 1, 2).astype(np.int32)


def _catmull_rom_spline(points: np.ndarray, num_output: int = 300) -> np.ndarray:
    """
    使用 Catmull-Rom 样条对轮廓点进行平滑插值，生成连续光滑曲线。

    Catmull-Rom 样条穿过所有控制点，且在各段之间 C1 连续，
    能有效消除多边形锯齿，产生自然平滑的曲线。

    参数:
        points: 控制点数组 (N, 2)
        num_output: 输出点数（越大越平滑）

    返回:
        平滑曲线点数组 (num_output, 2) int32
    """
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return pts.astype(np.int32)

    # 闭合曲线：在首尾各添加一个点
    pts = np.vstack([pts[-1:], pts, pts[:2]])

    result = []
    # 每段插值点数
    seg_points = max(1, num_output // n)

    for i in range(1, n + 1):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for t in np.linspace(0, 1, seg_points, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            # Catmull-Rom 基矩阵
            x = 0.5 * (
                (2 * p1[0]) +
                (-p0[0] + p2[0]) * t +
                (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1]) +
                (-p0[1] + p2[1]) * t +
                (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            result.append([x, y])

    result = np.array(result, dtype=np.int32)
    return result


def _smooth_and_resample_contour(contour: np.ndarray, num_points: int = 200,
                                  smooth_iters: int = 5) -> np.ndarray:
    """
    综合轮廓平滑：先拉普拉斯平滑去锯齿，再 Catmull-Rom 样条重采样。

    参数:
        contour: OpenCV 轮廓 (N, 1, 2)
        num_points: 最终输出点数
        smooth_iters: 拉普拉斯平滑迭代次数

    返回:
        平滑后的轮廓 (M, 1, 2) int32
    """
    pts = contour.reshape(-1, 2)

    # Step 1: 拉普拉斯平滑（去锯齿），alpha=0.15 轻度平滑避免收缩
    smoothed = _laplacian_smooth_contour(contour, iterations=smooth_iters, alpha=0.15)
    pts_smooth = smoothed.reshape(-1, 2)

    # Step 2: 如果点数太多，先降采样以减少样条计算量
    if len(pts_smooth) > 80:
        # 等弧长降采样到 60-80 个控制点
        indices = np.linspace(0, len(pts_smooth) - 1, 70, dtype=int)
        pts_smooth = pts_smooth[indices]

    # Step 3: 如果点数太少，直接返回拉普拉斯平滑结果
    if len(pts_smooth) < 4:
        return pts_smooth.reshape(-1, 1, 2).astype(np.int32)

    # Step 4: Catmull-Rom 样条插值
    spline_pts = _catmull_rom_spline(pts_smooth, num_output=num_points)

    return spline_pts.reshape(-1, 1, 2)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    填充二值掩膜中的孔洞（使用轮廓填充法）。
    """
    if mask.sum() == 0:
        return mask
    # 找到所有轮廓（含内孔）
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if hierarchy is not None:
        for i, cnt in enumerate(cnts):
            # 填充外轮廓（层级为 0 或无父轮廓）
            if hierarchy[0][i][3] == -1:
                cv2.fillPoly(filled, [cnt], 255)
    else:
        cv2.fillPoly(filled, cnts, 255)
    return filled


def _detect_skin_ycbcr(img_array: np.ndarray) -> np.ndarray:
    """
    使用 YCbCr 色彩空间检测皮肤区域。

    皮肤在 YCbCr 空间有较稳定的分布：
    - Cb: 77-127
    - Cr: 133-173

    参数:
        img_array: RGB 图像数组

    返回:
        皮肤区域布尔掩码
    """
    ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    cb = ycrcb[:, :, 1]
    cr = ycrcb[:, :, 2]
    # 皮肤色域（YCbCr 标准）
    skin_mask = (cb >= 77) & (cb <= 130) & (cr >= 135) & (cr <= 175)
    return skin_mask


# ============================================================
# U2-Net 分割后端（基于 rembg，方案 A）
# ============================================================

# 全局 U2-Net 会话缓存（避免每次推理都重新加载模型）
_U2NET_SESSION = None


def _get_u2net_session():
    """
    获取（必要时创建）U2-Net 推理会话。
    首次调用会加载模型（约 176MB），后续调用复用缓存。

    返回:
        rembg session 对象，或 None（如果 rembg 不可用）
    """
    global _U2NET_SESSION
    if _U2NET_SESSION is not None:
        return _U2NET_SESSION
    try:
        from rembg import new_session
        _U2NET_SESSION = new_session("u2net")
        return _U2NET_SESSION
    except Exception:
        return None


def segment_tongue_u2net(image_input) -> dict:
    """
    使用 U2-Net (via rembg) 进行舌体分割，采用"U2-Net ROI + HSV 颜色精筛"混合策略。

    设计理念：U2-Net 是通用显著性分割模型，会把嘴唇/面颊等显著物体一并纳入
    mask。因此不直接用 U2-Net mask 作为最终结果，而是把它当作"舌体大致位置"
    的 ROI，再在 ROI 内用严格的 HSV 颜色过滤提取真正的舌体像素。

    流程:
        1. 加载并预处理图像（与 segment_tongue 一致，max_size=512）
        2. rembg U2-Net 推理获取 alpha mask 作为 ROI
        3. 后处理（混合策略）：
           a. 轻度膨胀 ROI（1 像素，弥补 U2-Net 边缘内缩）
           b. 在 ROI 内做严格舌体颜色过滤（红/紫/淡粉色调 + 红色优势 > 5）
           c. YCbCr 皮肤检测，排除残留皮肤像素（红色优势 < 8 的皮肤扣除）
           d. 形态学闭运算 + 填充孔洞
           e. 连通域分析保留最大区域
           f. 高宽比约束（排除下巴但保留舌尖）
           g. 拉普拉斯 + Catmull-Rom 样条边界平滑
        4. 形状验证 + 最终轮廓提取

    相比纯 U2-Net：解决过分割（含嘴唇/面颊）和欠分割（舌根）问题
    相比纯 HSV：利用 U2-Net 的定位能力提高召回率

    参数:
        image_input: PIL.Image / numpy数组 / 文件路径

    返回:
        dict 包含:
            - mask: 舌体掩膜（与原图同尺寸，白色为舌体区域）
            - masked_image: 仅保留舌体区域的图像（背景为黑）
            - contour_image: 带轮廓标注的图像
            - tongue_region: 舌体区域的裁剪图像
            - coverage: 舌体面积占图像比例
            - success: 是否成功分割
            - resized_array: resize 后的图像数组（与 mask 同尺寸）
            - backend: 使用的后端名称（"u2net" / "u2net_unavailable" / ...）
    """
    # ===== 阶段0: 加载并预处理图像（保持与 segment_tongue 一致） =====
    img = load_image(image_input)
    img = resize_image(img, max_size=512)
    img_array = np.array(img)
    h_img, w_img = img_array.shape[:2]
    total_area = h_img * w_img

    contour_image = img_array.copy()
    coverage = 0.0
    success = False
    clean_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    # ===== 阶段1: 检查 rembg 可用性 =====
    try:
        from rembg import remove
    except ImportError:
        return {
            "mask": clean_mask,
            "masked_image": np.zeros_like(img_array),
            "contour_image": contour_image,
            "tongue_region": img_array.copy(),
            "coverage": 0.0,
            "success": False,
            "resized_array": img_array,
            "backend": "u2net_unavailable",
        }

    session = _get_u2net_session()
    if session is None:
        return {
            "mask": clean_mask,
            "masked_image": np.zeros_like(img_array),
            "contour_image": contour_image,
            "tongue_region": img_array.copy(),
            "coverage": 0.0,
            "success": False,
            "resized_array": img_array,
            "backend": "u2net_session_failed",
        }

    # ===== 阶段2: U2-Net 推理 =====
    try:
        pil_img = Image.fromarray(img_array)
        result = remove(pil_img, session=session)

        # result 可能是 PIL.Image 或 bytes
        if isinstance(result, Image.Image):
            result_array = np.array(result)
        else:
            result_array = cv2.imdecode(
                np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED
            )

        if result_array is None:
            raise RuntimeError("rembg 返回结果解码失败")

        # 提取 alpha 通道作为 mask
        if result_array.ndim == 3 and result_array.shape[2] == 4:
            alpha = result_array[:, :, 3]
        elif result_array.ndim == 3 and result_array.shape[2] == 3:
            # 没有 alpha，用灰度反相作为近似 mask
            alpha = cv2.cvtColor(result_array, cv2.COLOR_RGB2GRAY)
        else:
            alpha = result_array

        # 二值化 alpha（阈值 127）
        _, u2net_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        return {
            "mask": clean_mask,
            "masked_image": np.zeros_like(img_array),
            "contour_image": contour_image,
            "tongue_region": img_array.copy(),
            "coverage": 0.0,
            "success": False,
            "resized_array": img_array,
            "backend": "u2net_error",
            "error": str(e),
        }

    # 确保 mask 尺寸与原图一致
    if u2net_mask.shape[:2] != (h_img, w_img):
        u2net_mask = cv2.resize(u2net_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

    # 空 mask 直接返回失败
    if u2net_mask.sum() == 0:
        return {
            "mask": clean_mask,
            "masked_image": np.zeros_like(img_array),
            "contour_image": contour_image,
            "tongue_region": img_array.copy(),
            "coverage": 0.0,
            "success": False,
            "resized_array": img_array,
            "backend": "u2net_empty_mask",
        }

    # ===== 阶段3: 后处理流水线（混合策略：U2-Net 作 ROI + HSV 颜色精筛） =====
    # 设计理念：U2-Net 是通用显著性分割，会把嘴唇/面颊一并纳入。
    # 因此把 U2-Net mask 当作"舌体大致位置"的 ROI，再用严格的颜色过滤
    # 在 ROI 内提取真正的舌体像素。这样既利用了 U2-Net 的定位能力，
    # 又用颜色特异性排除嘴唇/皮肤，避免过分割。

    clean_mask = u2net_mask.copy()

    # Step 3a: 轻度膨胀 ROI（1 像素，仅弥补 U2-Net 边缘内缩，不再激进扩张）
    kernel_light = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_mask = cv2.dilate(clean_mask, kernel_light, iterations=1)

    # Step 3b: 在 ROI 内做自适应舌体颜色过滤
    # 舌体颜色特征：红色优势 R - max(G, B) > 阈值（嘴唇通常 0-4，皮肤通常 < 3）
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    red_adv = r - np.maximum(g, b)

    # 舌体颜色候选（在 ROI 内）：
    # - 红色调：hue < 25 或 hue > 155，sat > 10，val > 40
    # - 紫色调（青紫舌）：hue 115-175，sat > 5
    # - 淡白/粉色：red_adv > 0 且 sat < 50
    red_hue = (hue < 25) | (hue > 155)
    purple_hue = (hue > 115) & (hue < 175)
    pale_pink = (red_adv > 0) & (sat < 50) & (val > 80)

    # YCbCr 皮肤检测
    skin_mask = _detect_skin_ycbcr(img_array)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 自适应阈值：U2-Net 过分割时（ROI 覆盖率高），逐步收紧 red_adv 阈值
    # 从宽松到严格：3 → 8 → 15 → 20 → 30 → 40，选择使覆盖率 < 0.45 的最低阈值
    # 保证正常图片用宽松阈值（不丢淡白舌），过分割图片自动收紧
    red_adv_thresholds = [3, 8, 15, 20, 30, 40]
    best_clean_mask = None

    for red_thresh in red_adv_thresholds:
        tongue_color_mask = (
            ((red_hue & (sat > 10) & (val > 40)) | purple_hue | pale_pink)
            & (red_adv > red_thresh)
            & (val < 245)    # 排除高光过曝
        )

        # 在 ROI 内取交集
        test_mask = np.where(roi_mask > 0, tongue_color_mask.astype(np.uint8) * 255, 0).astype(np.uint8)

        # YCbCr 皮肤排除：排除红色优势低于当前阈值的皮肤像素
        skin_to_remove = skin_mask & (red_adv < red_thresh + 2)
        test_mask[skin_to_remove] = 0

        # 形态学清理（闭运算 + 填充孔洞）
        test_mask = cv2.morphologyEx(test_mask, cv2.MORPH_CLOSE, kernel_close)
        test_mask = _fill_holes(test_mask)

        # 连通域分析，保留最大区域（去除离散斑块）
        num_labels_t, labels_t, stats_t, _ = cv2.connectedComponentsWithStats(
            test_mask, connectivity=8
        )
        if num_labels_t > 2:
            max_label_t = 1 + np.argmax(stats_t[1:, cv2.CC_STAT_AREA])
            test_mask = np.where(labels_t == max_label_t, 255, 0).astype(np.uint8)

        test_cov = test_mask.sum() / 255 / total_area

        # 形状有效性检查（长宽比 0.25-2.8）
        shape_ok = False
        if test_cov > 0.005:
            cnts_t, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_t:
                cnt_t = max(cnts_t, key=cv2.contourArea)
                x_t, y_t, w_t, h_t = cv2.boundingRect(cnt_t)
                aspect_t = w_t / max(h_t, 1)
                shape_ok = (0.25 < aspect_t < 2.8)

        if test_cov > 0.005 and shape_ok:  # mask 不为空且形状有效
            best_clean_mask = test_mask
            if test_cov < 0.45:
                break  # 覆盖率合理，停止收紧

    # 使用自适应过滤的最佳结果
    if best_clean_mask is not None:
        clean_mask = _smooth_mask(best_clean_mask, blur_ksize=5)
    else:
        # 防御：所有阈值下 mask 都为空，回退到 U2-Net 原始 mask
        clean_mask = u2net_mask.copy()
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
        clean_mask = _fill_holes(clean_mask)
        clean_mask = _smooth_mask(clean_mask, blur_ksize=5)

    # Step 3f: 高宽比约束（排除下巴但保留舌尖，与 HSV 后端一致）
    ys, xs = np.where(clean_mask > 0)
    if len(ys) > 0:
        y_top, y_bottom = ys.min(), ys.max()
        x_left, x_right = xs.min(), xs.max()
        mask_w = x_right - x_left
        mask_h = y_bottom - y_top
        # 舌体高度通常不超过宽度的 1.5 倍；过高说明含下巴
        if mask_w > 0 and mask_h > mask_w * 1.5:
            max_height = int(mask_w * 1.3)
            new_bottom = y_top + max_height
            if y_bottom > new_bottom:
                clean_mask[new_bottom:, :] = 0
                clean_mask = _fill_holes(clean_mask)

    # ===== 阶段4: 提取最终轮廓 + 平滑 =====
    final_cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if final_cnts:
        final_cnt = max(final_cnts, key=cv2.contourArea)
        final_area = cv2.contourArea(final_cnt)
        coverage = final_area / total_area

        # 形状验证：长宽比 0.25-2.8 之间
        fx, fy, fw, fh = cv2.boundingRect(final_cnt)
        final_aspect = fw / max(fh, 1)
        shape_valid = (0.25 < final_aspect < 2.8)

        if 0.005 < coverage < 0.85 and shape_valid:
            # 边界平滑：拉普拉斯 + Catmull-Rom 样条
            peri = cv2.arcLength(final_cnt, True)
            approx = cv2.approxPolyDP(final_cnt, 0.006 * peri, True)
            final_cnt = _smooth_and_resample_contour(
                approx, num_points=200, smooth_iters=6
            )

            # 重新用平滑轮廓填充 mask
            clean_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(clean_mask, [final_cnt], 255)

            masked_image = cv2.bitwise_and(img_array, img_array, mask=clean_mask)
            cv2.drawContours(contour_image, [final_cnt], -1, (0, 255, 0), 3, cv2.LINE_AA)

            x, y, w, h = cv2.boundingRect(final_cnt)
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            tongue_region = img_array[y1:y2, x1:x2]

            success = True
        else:
            masked_image = np.zeros_like(img_array)
            tongue_region = img_array.copy()
    else:
        masked_image = np.zeros_like(img_array)
        tongue_region = img_array.copy()

    return {
        "mask": clean_mask,
        "masked_image": masked_image,
        "contour_image": contour_image,
        "tongue_region": tongue_region,
        "coverage": float(coverage),
        "success": success,
        "resized_array": img_array,
        "backend": "u2net",
    }


def segment_tongue(image_input, backend: str = "hsv") -> dict:
    """
    从图像中分割出舌体区域。

    流程：
    1. 加载并预处理图像
    2. 根据 backend 选择分割后端：
       - "hsv": HSV 色彩阈值法（多色相候选 + 位置先验 + 形状验证 + 边缘精修）
       - "u2net": U2-Net 深度学习分割（U2-Net ROI + HSV 颜色精筛混合策略）
    3. 形状验证 + 最终轮廓提取

    参数:
        image_input: PIL.Image / numpy数组 / 文件路径
        backend: 分割后端选择，可选值：
            - "hsv"（默认）: HSV 色彩阈值法
              精度高（不包含嘴唇/面颊），但可能欠分割（边缘覆盖不全）
            - "u2net": U2-Net 深度学习分割
              召回率高（覆盖全舌体），但可能含嘴唇/面颊（通用显著性分割）
              若 rembg 未安装或推理异常，自动回退到 HSV

    返回:
        dict 包含:
            - mask: 舌体掩膜（与原图同尺寸，白色为舌体区域）
            - masked_image: 仅保留舌体区域的图像（背景为黑）
            - contour_image: 带轮廓标注的图像
            - tongue_region: 舌体区域的裁剪图像
            - coverage: 舌体面积占图像比例
            - success: 是否成功分割
            - resized_array: resize 后的图像数组（与 mask 同尺寸）
            - backend: 实际使用的后端名称
              （"hsv" / "u2net" / "u2net_fallback" / "u2net_unavailable"）
    """
    # ===== 二选一策略：由用户指定后端，不再自动兜底 =====
    # 设计理由：
    #   - HSV 几乎不会"失败"（success=True），但可能识别不准确（欠分割）
    #   - 原"HSV 优先 + U2-Net 兜底"策略导致 U2-Net 形同虚设
    #   - 因此改为用户手动选择后端，让人决定使用哪种方式进行分析
    if backend == "u2net":
        try:
            u2net_result = segment_tongue_u2net(image_input)
            # 若 U2-Net 不可用（rembg 未安装等），返回结果已标记 backend，
            # 此时回退到 HSV
            if u2net_result.get("backend") in ("u2net_unavailable", "u2net_session_failed"):
                print(f"[TongueAI] U2-Net 不可用（{u2net_result['backend']}），回退到 HSV")
                hsv_result = _segment_tongue_hsv(image_input)
                hsv_result["backend"] = "u2net_fallback"
                return hsv_result
            u2net_result["backend"] = "u2net"
            return u2net_result
        except Exception as e:
            print(f"[TongueAI] U2-Net 分割异常，回退到 HSV: {e}")
            hsv_result = _segment_tongue_hsv(image_input)
            hsv_result["backend"] = "u2net_fallback"
            return hsv_result
    else:
        # 默认使用 HSV 色彩阈值法
        hsv_result = _segment_tongue_hsv(image_input)
        hsv_result["backend"] = "hsv"
        return hsv_result


def _segment_tongue_hsv(image_input) -> dict:
    """
    HSV 色彩阈值法舌体分割（原 segment_tongue 的核心实现）。

    这是项目原有的分割算法，基于多色相候选 + 位置先验 + 形状验证 +
    边缘精修。精度高（不包含嘴唇/面颊），但可能欠分割（边缘覆盖不全）。
    """
    # ===== 阶段0: 加载并预处理图像 =====
    img = load_image(image_input)
    img = resize_image(img, max_size=512)
    img_array = np.array(img)
    h_img, w_img = img_array.shape[:2]
    total_area = h_img * w_img

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    # ===== 阶段1: 多色相舌体候选检测（加宽阈值，提高覆盖率） =====
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)

    # --- 1.1 红色调（正常淡红舌、红舌、绛舌）- 加宽阈值 ---
    red_hue_mask = (hue < 20) | (hue > 160)
    red_mask = red_hue_mask & (sat > 10) & (val > 50) & (val < 240) & (r > g) & (r > b)

    # --- 1.2 紫色调（青紫舌、淡紫舌）- 加宽范围 ---
    purple_mask = (hue > 115) & (hue < 175) & (sat > 5) & (val > 40) & (val < 240)

    # --- 1.3 粉色调（淡白舌偏粉）- 放宽饱和度 ---
    pink_mask = (r > g) & (r > b) & (r - g > 3) & (r - b > 3) & (sat < 50) & (val > 80) & (val < 230)

    # --- 1.4 暗红/绛舌 ---
    dark_red_mask = (red_hue_mask | (hue > 170)) & (sat > 25) & (val > 35) & (val < 140) & (r > g + 8)

    # --- 1.5 淡白舌（明度高、红色微弱优势） ---
    pale_mask = (r > g) & (r >= b) & (r - g > 0) & (sat < 30) & (val > 100) & (val < 230) & (r > 100)

    # 合并所有舌体候选
    tongue_candidate_mask = red_mask | purple_mask | pink_mask | dark_red_mask | pale_mask

    # ===== 阶段2: 严格排除 UI 区域 =====
    # 2.1 排除橙色 UI 按钮
    orange_ui_mask = (hue > 3) & (hue < 35) & (sat > 60) & (val > 170)
    # 2.2 排除高亮白色 UI
    white_ui_mask = (val > 235) & (sat < 15)
    # 2.3 排除底部 UI 区域
    bottom_ui_mask = np.zeros((h_img, w_img), dtype=bool)
    bottom_ui_mask[int(h_img * 0.88):, :] = True
    # 2.4 排除顶部 UI 标题栏
    top_ui_mask = np.zeros((h_img, w_img), dtype=bool)
    top_ui_mask[:int(h_img * 0.05), :] = True

    exclude_mask = orange_ui_mask | white_ui_mask | bottom_ui_mask | top_ui_mask
    tongue_candidate_mask = tongue_candidate_mask & ~exclude_mask

    # ===== 阶段3: 自适应位置检测（替代固定位置先验） =====
    # 不再假设舌体在图像 62% 高度处，而是从候选像素中自适应定位
    # 策略：先做形态学清理 → 找连通域 → 按颜色+面积选择最佳中心 → 宽高斯加权

    # 先做轻度形态学清理，得到更连贯的候选区域
    candidate_uint8_temp = (tongue_candidate_mask.astype(np.uint8)) * 255
    kernel_temp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_uint8_temp = cv2.morphologyEx(candidate_uint8_temp, cv2.MORPH_CLOSE, kernel_temp)
    candidate_uint8_temp = cv2.morphologyEx(candidate_uint8_temp, cv2.MORPH_OPEN, kernel_temp)

    # 找到所有连通域，计算质心和颜色特征
    num_labels_temp, labels_temp, stats_temp, centroids_temp = cv2.connectedComponentsWithStats(
        candidate_uint8_temp, connectivity=8
    )

    adaptive_center = None
    if num_labels_temp > 1:
        min_area = max(50, total_area * 0.002)  # 至少 0.2% 或 50 像素
        valid_components = []
        for i in range(1, num_labels_temp):
            area = stats_temp[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            cx, cy = centroids_temp[i]
            # 颜色得分：该区域的平均颜色是否像舌体
            comp_pixels = img_array[labels_temp == i]
            if len(comp_pixels) == 0:
                continue
            mean_r = float(comp_pixels[:, 0].mean())
            mean_g = float(comp_pixels[:, 1].mean())
            mean_b = float(comp_pixels[:, 2].mean())
            # 舌体特征：R > G, R > B，且亮度适中
            color_score = 0.0
            if mean_r > mean_g and mean_r > mean_b:
                color_score = (mean_r - mean_g) + (mean_r - mean_b)
            # 亮度惩罚：过暗或过亮不是舌体
            brightness = (mean_r + mean_g + mean_b) / 3
            if brightness < 40 or brightness > 235:
                color_score *= 0.3
            valid_components.append({
                "centroid": (float(cx), float(cy)),
                "area": float(area),
                "color_score": float(color_score),
                "mean_r": mean_r,
            })

        if valid_components:
            # 按颜色得分 * log(面积) 排序，选择最佳中心
            # 舌体通常是面积较大且颜色偏红的区域
            valid_components.sort(
                key=lambda c: c["color_score"] * np.log1p(c["area"]), reverse=True
            )
            adaptive_center = valid_components[0]["centroid"]

    if adaptive_center is not None:
        center_x, center_y = adaptive_center
        # 用自适应中心构建位置先验（宽 sigma，允许较大范围）
        y_coords, x_coords = np.mgrid[0:h_img, 0:w_img]
        sigma_x = w_img * 0.35  # 更宽的 sigma，适应截图等不同场景
        sigma_y = h_img * 0.30
        position_weight = np.exp(
            -((x_coords - center_x) ** 2 / (2 * sigma_x ** 2) +
              (y_coords - center_y) ** 2 / (2 * sigma_y ** 2))
        )
        position_mask = position_weight > 0.15  # 更低阈值，保留更多候选
        tongue_candidate_mask = tongue_candidate_mask & position_mask
    else:
        # 无候选中心时，使用图像中央作为默认中心
        center_x, center_y = w_img * 0.5, h_img * 0.5

    # ===== 阶段4: 形态学清理 =====
    candidate_uint8 = (tongue_candidate_mask.astype(np.uint8)) * 255
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 开运算去噪
    candidate_uint8 = cv2.morphologyEx(candidate_uint8, cv2.MORPH_OPEN, kernel_small)
    # 闭运算填洞
    candidate_uint8 = cv2.morphologyEx(candidate_uint8, cv2.MORPH_CLOSE, kernel_med)

    # ===== 阶段5: 连通区域分析 + 形状评分 =====
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_uint8, connectivity=8)

    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cov = area / total_area

        # 面积过滤（放宽下限以适应截图等小舌体场景）
        if cov < 0.005 or cov > 0.40:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        # 提取轮廓
        region_mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # 形状特征
        aspect_ratio = cw / max(ch, 1)  # 长宽比（舌体 0.6-2.5）
        fill_ratio = area / max(cw * ch, 1)  # 填充率（舌体 0.50-0.95）

        # 位置得分（越接近自适应中心越好，但权重降低）
        pos_dist = np.sqrt(((cx - center_x) / w_img) ** 2 + ((cy - center_y) / h_img) ** 2)
        position_score = np.exp(-pos_dist * 2.0)

        # 面积得分（放宽范围，适应截图场景）
        if 0.05 <= cov <= 0.25:
            area_score = 1.0
        elif 0.02 <= cov <= 0.35:
            area_score = 0.7
        elif 0.008 <= cov:
            area_score = 0.5
        else:
            area_score = 0.2

        # 长宽比得分（放宽范围，舌体可能较窄或较宽）
        if 0.6 <= aspect_ratio <= 2.2:
            ar_score = 1.0
        elif 0.4 <= aspect_ratio <= 2.8:
            ar_score = 0.6
        else:
            ar_score = 0.2

        # 填充率得分
        if 0.50 <= fill_ratio <= 0.95:
            fr_score = 1.0
        elif 0.35 <= fill_ratio <= 0.98:
            fr_score = 0.7
        else:
            fr_score = 0.3

        # 颜色得分（提高权重，更细致评估）
        region_pixels = img_array[region_mask > 0]
        if len(region_pixels) == 0:
            continue
        mean_rgb = region_pixels.mean(axis=0)
        mr, mg, mb = mean_rgb
        # 舌体颜色：R 通常 >= G, B
        color_red_dominant = (mr > mg) & (mr > mb)
        color_reasonable_brightness = (mr > 60) & (mr < 235)
        # 细化颜色得分：红色优势越大越好
        red_advantage = (mr - mg) + (mr - mb)
        if color_red_dominant and color_reasonable_brightness:
            color_score = min(1.0, 0.5 + red_advantage / 60.0)
        elif color_reasonable_brightness:
            color_score = 0.4
        else:
            color_score = 0.1

        # 综合得分（颜色权重提高，位置权重降低）
        total_score = (
            position_score * 0.20 +
            area_score * 0.15 +
            ar_score * 0.15 +
            fr_score * 0.15 +
            color_score * 0.35
        )

        candidates.append({
            "contour": cnt,
            "mask": region_mask,
            "area": area,
            "coverage": cov,
            "bbox": (x, y, cw, ch),
            "centroid": (cx, cy),
            "aspect_ratio": aspect_ratio,
            "fill_ratio": fill_ratio,
            "score": total_score,
            "mean_rgb": mean_rgb,
        })

    # 多候选合并：如果多个候选区域距离较近，合并它们
    if len(candidates) > 1:
        candidates.sort(key=lambda c: -c["score"])
        best_c = candidates[0]
        merged = [best_c]
        for c in candidates[1:]:
            # 计算与最佳候选的距离
            dist = np.sqrt(
                ((c["centroid"][0] - best_c["centroid"][0]) / w_img) ** 2 +
                ((c["centroid"][1] - best_c["centroid"][1]) / h_img) ** 2
            )
            # 如果距离近且面积合理，合并
            if dist < 0.15 and c["coverage"] > 0.003:
                merged.append(c)
        # 如果有多个待合并区域，创建合并掩膜
        if len(merged) > 1:
            merged_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            for c in merged:
                merged_mask = cv2.bitwise_or(merged_mask, c["mask"])
            # 对合并后的掩膜做闭运算连接
            merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            # 将合并掩膜作为最佳候选
            cnts_m, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_m:
                best_cnt_m = max(cnts_m, key=cv2.contourArea)
                merged_area = cv2.contourArea(best_cnt_m)
                merged_cov = merged_area / total_area
                if 0.005 < merged_cov < 0.40:
                    best_c["mask"] = merged_mask
                    best_c["contour"] = best_cnt_m
                    best_c["area"] = merged_area
                    best_c["coverage"] = merged_cov
                    candidates[0] = best_c

    # ===== 阶段6: 选择最佳候选并精修 =====
    success = False
    contour_image = img_array.copy()
    tongue_region = img_array.copy()
    coverage = 0.0
    clean_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    if candidates:
        # 按综合得分排序
        candidates.sort(key=lambda c: -c["score"])
        best = candidates[0]

        # 阈值检查：最佳候选得分必须 > 0.30（调低以适应截图场景）
        if best["score"] > 0.30:
            best_cnt = best["contour"]

            # === 综合方案：实际掩膜 + 连通域 + 自适应下巴排除 + 轻度扩边 + 平滑 ===
            # 保留自然舌形（无凸包）
            # 自适应下边界（根据掩膜几何，非固定截断）
            # 连通域分析（去除不连续下巴斑块）
            # 轻度膨胀（扩展舌侧）
            # 增强轮廓平滑

            # Step 1: 使用实际检测掩膜
            region_mask = best["mask"]

            # Step 2: 轻度形态学闭运算（填充小间隙）
            kernel_light = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel_light)

            # Step 3: 填充孔洞 + 掩膜平滑
            region_mask = _fill_holes(region_mask)
            region_mask = _smooth_mask(region_mask, blur_ksize=5)

            # Step 4: GrabCut 精修
            try:
                refined_mask = _grabcut_refine(img_array, region_mask)
                if refined_mask is not None:
                    refined_area = (refined_mask > 0).sum()
                    if 0.005 < refined_area / total_area < 0.40:
                        clean_mask = refined_mask
                    else:
                        clean_mask = region_mask
                else:
                    clean_mask = region_mask
            except Exception:
                clean_mask = region_mask

            # Step 5: 后处理
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_light)
            clean_mask = _fill_holes(clean_mask)
            clean_mask = _smooth_mask(clean_mask, blur_ksize=5)

            # Step 5.5: 皮肤排除（舌体比皮肤更红，排除低红色优势的像素）
            # 这一步防止宽颜色阈值把皮肤/下巴纳入舌体
            current_cov = (clean_mask > 0).sum() / total_area
            if current_cov > 0.06:  # 仅在覆盖率较高时执行（小区域不需要）
                mask_pixels = img_array[clean_mask > 0]
                if len(mask_pixels) > 0:
                    mr_ch = mask_pixels[:, 0].astype(np.float32)
                    mg_ch = mask_pixels[:, 1].astype(np.float32)
                    mb_ch = mask_pixels[:, 2].astype(np.float32)
                    # 红色优势：R - max(G, B)
                    red_adv = mr_ch - np.maximum(mg_ch, mb_ch)
                    # 舌体像素的红色优势通常 > 5，皮肤通常 < 5
                    # 如果覆盖率很高（>12%），提高阈值以排除更多皮肤
                    red_thresh = 8 if current_cov > 0.12 else 3
                    # 创建皮肤排除掩膜
                    skin_in_mask = red_adv < red_thresh
                    # 仅排除皮肤像素，保留舌体像素
                    mask_indices = np.where(clean_mask > 0)
                    skin_exclusion = np.zeros_like(clean_mask)
                    skin_exclusion[mask_indices[0][skin_in_mask],
                                   mask_indices[1][skin_in_mask]] = 255
                    # 从 clean_mask 中减去皮肤区域
                    clean_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(skin_exclusion))
                    # 闭运算修复减除后的间隙
                    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_light)
                    clean_mask = _fill_holes(clean_mask)
                    clean_mask = _smooth_mask(clean_mask, blur_ksize=5)

            # Step 5.6: 最大覆盖率保护（如果仍然过高，用更严格的红度阈值）
            current_cov = (clean_mask > 0).sum() / total_area
            if current_cov > 0.20:
                # 覆盖率过高，使用更严格的红度阈值重新过滤
                mask_pixels = img_array[clean_mask > 0]
                if len(mask_pixels) > 0:
                    mr_ch = mask_pixels[:, 0].astype(np.float32)
                    mg_ch = mask_pixels[:, 1].astype(np.float32)
                    mb_ch = mask_pixels[:, 2].astype(np.float32)
                    red_adv = mr_ch - np.maximum(mg_ch, mb_ch)
                    # 更严格阈值：只保留红色优势 > 12 的像素
                    skin_in_mask = red_adv < 12
                    mask_indices = np.where(clean_mask > 0)
                    skin_exclusion = np.zeros_like(clean_mask)
                    skin_exclusion[mask_indices[0][skin_in_mask],
                                   mask_indices[1][skin_in_mask]] = 255
                    clean_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(skin_exclusion))
                    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_light)
                    clean_mask = _fill_holes(clean_mask)
                    clean_mask = _smooth_mask(clean_mask, blur_ksize=5)

            # Step 6: 连通域分析（保留最大连通域，去除不连续的下巴斑块）
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
            if num_labels > 2:  # 背景 + 多个前景
                # 找最大前景连通域（跳过背景 label 0）
                max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                clean_mask = np.where(labels == max_label, 255, 0).astype(np.uint8)

            # Step 7: 自适应下边界 + 高宽比约束（排除下巴但保留舌尖）
            ys, xs = np.where(clean_mask > 0)
            if len(ys) > 0:
                y_top = ys.min()
                y_bottom = ys.max()
                x_left = xs.min()
                x_right = xs.max()
                mask_w = x_right - x_left
                mask_h = y_bottom - y_top
                y_center = int(np.mean(ys))
                mask_height_above = y_center - y_top  # 质心到顶部的距离

                # 7a: 高宽比约束（舌体高度通常不超过宽度的 1.5 倍）
                # 如果掩膜过高，说明包含了下巴，需要截断
                if mask_w > 0 and mask_h > mask_w * 1.5:
                    # 按宽度 1.3 倍限制高度
                    max_height = int(mask_w * 1.3)
                    new_bottom = y_top + max_height
                    if y_bottom > new_bottom:
                        clean_mask[new_bottom:, :] = 0
                        clean_mask = _fill_holes(clean_mask)

                # 7b: 自适应下边界（质心 + 1.5 * 上半部分高度，收紧因子）
                ys, xs = np.where(clean_mask > 0)
                if len(ys) > 0:
                    y_top = ys.min()
                    y_bottom = ys.max()
                    y_center = int(np.mean(ys))
                    mask_height_above = y_center - y_top
                    adaptive_lower = y_center + int(1.5 * mask_height_above)
                    adaptive_lower = min(adaptive_lower, h_img)

                    if y_bottom > adaptive_lower:
                        clean_mask[adaptive_lower:, :] = 0
                        clean_mask = _fill_holes(clean_mask)

            # Step 8: 轻度膨胀（扩展舌侧，改善边缘覆盖）
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean_mask = cv2.dilate(clean_mask, kernel_dilate, iterations=1)

            # Step 9: 提取最终轮廓
            final_cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if final_cnts:
                final_cnt = max(final_cnts, key=cv2.contourArea)
                final_area = cv2.contourArea(final_cnt)
                coverage = final_area / total_area

                # 形状验证：计算边界框长宽比，排除文字条/UI 元素误检
                fx, fy, fw, fh = cv2.boundingRect(final_cnt)
                final_aspect = fw / max(fh, 1)
                # 舌体长宽比通常在 0.3-2.5 之间，超出范围可能是文字或 UI
                shape_valid = (0.25 < final_aspect < 2.8)

                if 0.005 < coverage < 0.45 and shape_valid:
                    # Step 10: 增强轮廓平滑（6次迭代，平衡平滑度与自然形状）
                    peri = cv2.arcLength(final_cnt, True)
                    approx = cv2.approxPolyDP(final_cnt, 0.006 * peri, True)
                    final_cnt = _smooth_and_resample_contour(
                        approx, num_points=200, smooth_iters=6
                    )

                    # 最终 mask 用平滑轮廓填充
                    clean_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    cv2.fillPoly(clean_mask, [final_cnt], 255)

                    masked_image = cv2.bitwise_and(img_array, img_array, mask=clean_mask)
                    cv2.drawContours(contour_image, [final_cnt], -1, (0, 255, 0), 3, cv2.LINE_AA)

                    x, y, w, h = cv2.boundingRect(final_cnt)
                    pad = 5
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
                    tongue_region = img_array[y1:y2, x1:x2]

                    success = True
                else:
                    masked_image = np.zeros_like(img_array)
            else:
                masked_image = np.zeros_like(img_array)
        else:
            masked_image = np.zeros_like(img_array)
    else:
        masked_image = np.zeros_like(img_array)

    # 兜底：如果多色相检测失败，回退到原始位置先验+区域生长法
    used_fallback = False
    if not success:
        fallback_result = _fallback_region_growing(img_array, hsv, lab, h_img, w_img, total_area, exclude_mask)
        if fallback_result is not None:
            clean_mask, masked_image, contour_image, tongue_region, coverage, success = fallback_result
            used_fallback = True

    return {
        "mask": clean_mask,
        "masked_image": masked_image,
        "contour_image": contour_image,
        "tongue_region": tongue_region,
        "coverage": float(coverage),
        "success": success,
        "resized_array": img_array,
        "backend": "hsv_fallback" if used_fallback else "hsv",
    }


def _grabcut_refine(img_array: np.ndarray, initial_mask: np.ndarray):
    """
    使用 GrabCut 算法精细化分割边界。

    GrabCut 通过迭代能量优化获得更精确的前景/背景分割。
    需要初始 mask 提供前景/背景的大致位置。

    参数:
        img_array: RGB 图像数组
        initial_mask: 初始二值掩膜（255=前景，0=背景）

    返回:
        精细化后的掩膜，或 None（如果 GrabCut 失败）
    """
    if initial_mask.sum() == 0:
        return None

    h, w = img_array.shape[:2]
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 初始化 GrabCut mask
    gc_mask = np.zeros((h, w), np.uint8)
    gc_mask[initial_mask > 0] = cv2.GC_PR_FGD
    gc_mask[initial_mask == 0] = cv2.GC_PR_BGD

    # 确定前景（腐蚀内部）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(initial_mask, kernel, iterations=2)
    gc_mask[eroded > 0] = cv2.GC_FGD

    # 确定背景（膨胀外的区域）
    dilated = cv2.dilate(initial_mask, kernel, iterations=4)
    bg_region = (dilated == 0)
    gc_mask[bg_region] = cv2.GC_BGD

    # 执行 GrabCut（增加迭代次数以提高精度）
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # 提取前景
    result_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # 形态学后处理
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel_close)

    # 验证结果有效性
    if result_mask.sum() == 0:
        return None

    return result_mask


def _fallback_region_growing(img_array, hsv, lab, h_img, w_img, total_area, exclude_mask):
    """
    兜底方案：自适应种子点区域生长法。
    当多色相检测失败时使用。
    不再限制固定搜索区域，而是在全图寻找最红的区域作为种子点。
    """
    # 在全图搜索最红的区域（舌体通常是最红的区域）
    r_full, g_full, b_full = (img_array[:, :, 0].astype(int),
                              img_array[:, :, 1].astype(int),
                              img_array[:, :, 2].astype(int))

    # 红色优势得分
    red_advantage = np.maximum(r_full - g_full, 0) + np.maximum(r_full - b_full, 0)
    brightness = (r_full + g_full + b_full) / 3
    brightness_score = np.exp(-((brightness - 128) ** 2) / (2 * 60 ** 2))

    # 排除橙色 UI
    orange_penalty = np.where(
        (r_full > 180) & (g_full > 80) & ((r_full - g_full) < 50), 0.1, 1.0
    )

    # 综合得分
    tongue_score = red_advantage * brightness_score * orange_penalty

    # 排除 UI 区域
    tongue_score[exclude_mask] = 0
    # 排除图像边缘（避免在边框处误检）
    edge_margin = 10
    tongue_score[:edge_margin, :] = 0
    tongue_score[-edge_margin:, :] = 0
    tongue_score[:, :edge_margin] = 0
    tongue_score[:, -edge_margin:] = 0

    # 找到得分最高的点作为种子
    max_idx = np.unravel_index(np.argmax(tongue_score), tongue_score.shape)
    seed_y, seed_x = int(max_idx[0]), int(max_idx[1])

    # 如果最高分太低，说明图像中没有明显的舌体
    if tongue_score[seed_y, seed_x] < 5:
        return None

    seed_lab = lab[seed_y, seed_x].astype(np.float32)

    # 区域生长
    diff = lab.astype(np.float32) - seed_lab
    color_dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # 自适应阈值
    local_region = lab[max(0, seed_y - 15):min(h_img, seed_y + 15),
                       max(0, seed_x - 15):min(w_img, seed_x + 15)]
    local_std = np.mean(np.std(local_region.astype(float), axis=(0, 1)))
    grow_threshold = max(20, min(45, local_std * 2.5 + 15))

    init_mask = (color_dist < grow_threshold).astype(np.uint8) * 255
    # 排除 UI 区域
    init_mask[exclude_mask] = 0
    # 不再限制固定高度区域（自适应种子点已定位舌体）

    # 形态学
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    init_mask = cv2.morphologyEx(init_mask, cv2.MORPH_OPEN, kernel)
    init_mask = cv2.morphologyEx(init_mask, cv2.MORPH_CLOSE, kernel)

    # 连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(init_mask, connectivity=8)
    best_cnt = None
    best_score = -1

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cov = area / total_area
        if cov < 0.005 or cov > 0.40:
            continue

        cx, cy = centroids[i]
        # 距种子点距离
        dist = np.sqrt((cx - seed_x) ** 2 + (cy - seed_y) ** 2)
        max_dist = np.sqrt(w_img ** 2 + h_img ** 2)
        position_score = 1.0 - (dist / max_dist)

        # 面积得分
        if 0.05 <= cov <= 0.25:
            area_score = 1.0
        elif 0.02 <= cov <= 0.35:
            area_score = 0.5
        else:
            area_score = 0.1

        total_score = position_score * 0.6 + area_score * 0.4
        if total_score > best_score:
            best_score = total_score
            region_mask = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                best_cnt = max(cnts, key=cv2.contourArea)

    if best_cnt is None or best_score < 0.2:
        return None

    # === 综合方案：实际掩膜 + 连通域 + 自适应下巴排除 + 轻度扩边 + 平滑 ===
    clean_mask = region_mask.copy()

    # 轻度形态学闭运算
    kernel_light_fb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_light_fb)
    clean_mask = _fill_holes(clean_mask)
    clean_mask = _smooth_mask(clean_mask, blur_ksize=5)

    try:
        refined = _grabcut_refine(img_array, clean_mask)
        if refined is not None:
            r_area = (refined > 0).sum()
            if 0.005 * total_area < r_area < 0.40 * total_area:
                clean_mask = refined
    except Exception:
        pass

    # 后处理
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_light_fb)
    clean_mask = _fill_holes(clean_mask)
    clean_mask = _smooth_mask(clean_mask, blur_ksize=5)

    # 皮肤排除（同主流程）
    current_cov_fb = (clean_mask > 0).sum() / total_area
    if current_cov_fb > 0.06:
        mask_pixels_fb = img_array[clean_mask > 0]
        if len(mask_pixels_fb) > 0:
            mr_fb = mask_pixels_fb[:, 0].astype(np.float32)
            mg_fb = mask_pixels_fb[:, 1].astype(np.float32)
            mb_fb = mask_pixels_fb[:, 2].astype(np.float32)
            red_adv_fb = mr_fb - np.maximum(mg_fb, mb_fb)
            red_thresh_fb = 8 if current_cov_fb > 0.12 else 3
            skin_in_fb = red_adv_fb < red_thresh_fb
            mask_idx_fb = np.where(clean_mask > 0)
            skin_excl_fb = np.zeros_like(clean_mask)
            skin_excl_fb[mask_idx_fb[0][skin_in_fb],
                         mask_idx_fb[1][skin_in_fb]] = 255
            clean_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(skin_excl_fb))
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_light_fb)
            clean_mask = _fill_holes(clean_mask)
            clean_mask = _smooth_mask(clean_mask, blur_ksize=5)

    # 连通域分析（保留最大连通域）
    num_labels_fb, labels_fb, stats_fb, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    if num_labels_fb > 2:
        max_label_fb = 1 + np.argmax(stats_fb[1:, cv2.CC_STAT_AREA])
        clean_mask = np.where(labels_fb == max_label_fb, 255, 0).astype(np.uint8)

    # 自适应下边界 + 高宽比约束
    ys_fb, xs_fb = np.where(clean_mask > 0)
    if len(ys_fb) > 0:
        y_top_fb = ys_fb.min()
        y_bot_fb = ys_fb.max()
        x_left_fb = xs_fb.min()
        x_right_fb = xs_fb.max()
        mask_w_fb = x_right_fb - x_left_fb
        mask_h_fb = y_bot_fb - y_top_fb
        y_center_fb = int(np.mean(ys_fb))
        mask_h_above_fb = y_center_fb - y_top_fb

        # 高宽比约束
        if mask_w_fb > 0 and mask_h_fb > mask_w_fb * 1.5:
            max_h_fb = int(mask_w_fb * 1.3)
            new_bot_fb = y_top_fb + max_h_fb
            if y_bot_fb > new_bot_fb:
                clean_mask[new_bot_fb:, :] = 0
                clean_mask = _fill_holes(clean_mask)

        # 自适应下边界（收紧因子 1.5）
        ys_fb, xs_fb = np.where(clean_mask > 0)
        if len(ys_fb) > 0:
            y_top_fb = ys_fb.min()
            y_center_fb = int(np.mean(ys_fb))
            mask_h_above_fb = y_center_fb - y_top_fb
            adaptive_lower_fb = min(y_center_fb + int(1.5 * mask_h_above_fb), h_img)
            if ys_fb.max() > adaptive_lower_fb:
                clean_mask[adaptive_lower_fb:, :] = 0
                clean_mask = _fill_holes(clean_mask)

    # 轻度膨胀
    kernel_dil_fb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.dilate(clean_mask, kernel_dil_fb, iterations=1)

    final_cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not final_cnts:
        return None

    final_cnt = max(final_cnts, key=cv2.contourArea)
    final_area = cv2.contourArea(final_cnt)
    coverage = final_area / total_area

    # 形状验证
    fbx, fby, fbw, fbh = cv2.boundingRect(final_cnt)
    fb_aspect = fbw / max(fbh, 1)
    fb_shape_valid = (0.25 < fb_aspect < 2.8)

    if not (0.005 < coverage < 0.45 and fb_shape_valid):
        return None

    # 增强轮廓平滑
    peri = cv2.arcLength(final_cnt, True)
    approx = cv2.approxPolyDP(final_cnt, 0.006 * peri, True)
    final_cnt = _smooth_and_resample_contour(approx, num_points=200, smooth_iters=6)

    clean_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(clean_mask, [final_cnt], 255)

    masked_image = cv2.bitwise_and(img_array, img_array, mask=clean_mask)
    contour_image = img_array.copy()
    # 抗锯齿绘制
    cv2.drawContours(contour_image, [final_cnt], -1, (0, 255, 0), 3, cv2.LINE_AA)

    x, y, w, h = cv2.boundingRect(final_cnt)
    pad = 5
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    tongue_region = img_array[y1:y2, x1:x2]

    return clean_mask, masked_image, contour_image, tongue_region, coverage, True


def analyze_tongue_body_color(image_array: np.ndarray, mask: np.ndarray) -> dict:
    """
    分析舌体区域的舌质颜色，判断舌质类型。

    基于 HSV 颜色特征进行分类：
    - 淡红舌：正常红润
    - 淡白舌：明度高、饱和度低
    - 红舌：红色占比高
    - 绛舌：深红色
    - 青紫舌：色调偏向紫色

    返回:
        dict 包含舌质类型键、名称、颜色特征等
    """
    from .knowledge import TONGUE_BODY_TYPES
    from .utils import compute_color_features

    features = compute_color_features(image_array, mask)

    mean_r, mean_g, mean_b = features["mean_rgb"]
    hue = features["mean_hue"]
    sat = features["mean_saturation"]
    val = features["mean_value"]
    redness = features["redness"]
    r_over_g = features["r_over_g"]

    # 分类逻辑（基于颜色特征的规则判断）
    # 注意：这是简化的规则分类，真实项目中应使用训练好的深度学习模型

    if hue > 250 or (hue > 120 and hue < 180 and sat > 30):
        # 色调偏紫蓝
        body_key = "purple"
    elif val > 75 and sat < 25 and redness < 0.38:
        # 明度高、饱和度低、红色占比低 → 淡白
        body_key = "pale"
    elif redness > 0.42 and r_over_g > 1.5 and val < 65:
        # 红色占比高、R/G比大、明度低 → 绛舌（深红）
        body_key = "crimson"
    elif redness > 0.40 and r_over_g > 1.35:
        # 红色占比高 → 红舌
        body_key = "red"
    else:
        # 默认为正常淡红舌
        body_key = "pale_red"

    body_info = TONGUE_BODY_TYPES[body_key]
    return {
        "body_key": body_key,
        "name": body_info["name"],
        "description": body_info["description"],
        "tcm_meaning": body_info["tcm_meaning"],
        "health_status": body_info["health_status"],
        "color_features": features,
    }


def analyze_coating_color(image_array: np.ndarray, mask: np.ndarray) -> dict:
    """
    分析舌苔颜色，判断舌苔类型。

    舌苔覆盖在舌体表面，通常呈白色、黄色或灰黑色。
    通过分析舌体区域内的颜色变异程度和色调来判断。

    返回:
        dict 包含舌苔类型键、名称等
    """
    from .knowledge import TONGUE_COATING_TYPES
    from .utils import compute_color_features

    features = compute_color_features(image_array, mask)

    mean_r, mean_g, mean_b = features["mean_rgb"]
    hue = features["mean_hue"]
    sat = features["mean_saturation"]
    val = features["mean_value"]
    std_rgb = features["std_rgb"]

    # 舌苔判断逻辑
    avg_std = np.mean(std_rgb)

    if hue > 30 and hue < 80 and sat > 20:
        # 黄色调
        if sat > 40 and avg_std > 30:
            coating_key = "greasy_yellow"
        else:
            coating_key = "yellow"
    elif avg_std > 45:
        # 颜色变异大 → 可能是剥苔
        coating_key = "peeled"
    elif val < 35 or (val < 50 and sat < 15):
        # 明度很低 → 灰黑苔
        coating_key = "gray_black"
    elif val > 65 and sat < 25:
        # 明度高、饱和度低 → 薄白苔或厚白苔
        if avg_std > 25:
            coating_key = "thick_white"
        else:
            coating_key = "thin_white"
    else:
        # 默认薄白苔
        coating_key = "thin_white"

    coating_info = TONGUE_COATING_TYPES[coating_key]
    return {
        "coating_key": coating_key,
        "name": coating_info["name"],
        "description": coating_info["description"],
        "tcm_meaning": coating_info["tcm_meaning"],
        "health_status": coating_info["health_status"],
        "color_features": features,
    }

