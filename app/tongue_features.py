"""
舌象21类特征分析模块
====================
基于 U2-Net/HSV 分割结果，按 YOLO 21 类标准分析舌象：

1. 舌质颜色（0-3）：健康舌、薄苔舌、红舌、紫舌
2. 舌体形态（4-5）：胖大舌、瘦舌
3. 舌面特征（6-8）：红点舌、裂纹舌、齿痕舌
4. 舌苔（9-12）：白苔、黄苔、黑苔、花苔
5. 舌面脏腑分区凹凸（13-20）：肾/肝胆/脾胃/心肺 区的凹陷与凸起

重要声明：本模块仅用于教育科普目的，不构成医疗诊断。
"""

import cv2
import numpy as np
from typing import Optional

from .atlas import YOLO_CLASSES, ORGAN_REGIONS, TONGUE_SHAPE_FEATURES


# ============================================================
# 1. 舌体形态分析（胖大舌 / 瘦舌）
# ============================================================

def analyze_tongue_shape(mask: np.ndarray, image_array: np.ndarray) -> dict:
    """
    通过掩膜轮廓分析舌体形态：胖大 / 瘦 / 正常。

    判断依据：
    - 胖大舌：宽高比偏大（舌体宽且厚），覆盖面积大
    - 瘦舌：宽高比偏小（舌体窄且薄），覆盖面积小

    参数:
        mask: 舌体二值掩膜
        image_array: 原始图像数组

    返回:
        {"shape": "pangda"/"shou"/"normal", "yolo_id": 4/5/None,
         "aspect_ratio": float, "coverage": float, "description": str}
    """
    h_img, w_img = mask.shape[:2]
    total_area = h_img * w_img

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"shape": "normal", "yolo_id": None, "aspect_ratio": 0, "coverage": 0,
                "description": "未检测到舌体轮廓"}

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    coverage = area / total_area

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / max(h, 1)

    # 舌体面积与边界框面积之比（填充率），反映舌体是否"饱满"
    fill_ratio = area / max(w * h, 1)

    # 胖大舌：宽高比大 + 覆盖面积大 + 填充率高
    if aspect_ratio > 1.15 and coverage > 0.10 and fill_ratio > 0.65:
        info = YOLO_CLASSES[4]
        return {"shape": "pangda", "yolo_id": 4, "aspect_ratio": round(aspect_ratio, 2),
                "coverage": round(coverage, 4), "fill_ratio": round(fill_ratio, 2),
                "description": info["tcm_meaning"]}

    # 瘦舌：宽高比小 + 覆盖面积小
    if (aspect_ratio < 0.72 or coverage < 0.035) and fill_ratio < 0.55:
        info = YOLO_CLASSES[5]
        return {"shape": "shou", "yolo_id": 5, "aspect_ratio": round(aspect_ratio, 2),
                "coverage": round(coverage, 4), "fill_ratio": round(fill_ratio, 2),
                "description": info["tcm_meaning"]}

    return {"shape": "normal", "yolo_id": None, "aspect_ratio": round(aspect_ratio, 2),
            "coverage": round(coverage, 4), "fill_ratio": round(fill_ratio, 2),
            "description": "舌体形态正常"}


# ============================================================
# 2. 红点舌检测
# ============================================================

def detect_red_spots(image_array: np.ndarray, mask: np.ndarray) -> dict:
    """
    检测舌面上的红色点状突起（蕈状乳头充血）。

    方法：在舌体区域内寻找局部红色异常高的像素簇。

    参数:
        image_array: RGB 图像数组
        mask: 舌体掩膜

    返回:
        {"detected": bool, "yolo_id": 6/None, "spot_count": int, "description": str}
    """
    if mask.sum() == 0:
        return {"detected": False, "yolo_id": None, "spot_count": 0, "description": "无舌体区域"}

    r = image_array[:, :, 0].astype(np.float32)
    g = image_array[:, :, 1].astype(np.float32)
    b = image_array[:, :, 2].astype(np.float32)

    # 红色优势：R 明显大于 G 和 B
    red_advantage = r - np.maximum(g, b)

    # 高红色优势像素（阈值 25，比舌体平均红色优势高很多）
    red_spot_mask = (red_advantage > 25) & (mask > 0) & (r > 150)

    # 形态学清理
    spot_uint8 = red_spot_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    spot_uint8 = cv2.morphologyEx(spot_uint8, cv2.MORPH_OPEN, kernel)
    spot_uint8 = cv2.morphologyEx(spot_uint8, cv2.MORPH_CLOSE, kernel)

    # 连通域分析，统计红点数量
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spot_uint8, connectivity=8)
    spot_count = 0
    min_spot_area = 8  # 最小红点面积
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_spot_area:
            spot_count += 1

    detected = spot_count >= 5  # 至少5个红点才算红点舌

    if detected:
        info = YOLO_CLASSES[6]
        return {"detected": True, "yolo_id": 6, "spot_count": spot_count,
                "description": info["tcm_meaning"]}

    return {"detected": False, "yolo_id": None, "spot_count": spot_count,
            "description": "未检测到明显红点"}


# ============================================================
# 3. 裂纹舌检测
# ============================================================

def detect_cracks(image_array: np.ndarray, mask: np.ndarray) -> dict:
    """
    检测舌面上的裂纹（线状凹陷）。

    方法：使用形态学 black-hat 操作检测暗色沟槽（裂纹比周围组织暗）。

    参数:
        image_array: RGB 图像数组
        mask: 舌体掩膜

    返回:
        {"detected": bool, "yolo_id": 7/None, "crack_score": float, "description": str}
    """
    if mask.sum() == 0:
        return {"detected": False, "yolo_id": None, "crack_score": 0, "description": "无舌体区域"}

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # 仅在舌体区域内分析
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # black-hat：提取比周围暗的细线（裂纹）
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    blackhat_h = cv2.morphologyEx(masked_gray, cv2.MORPH_BLACKHAT, kernel_rect)

    kernel_rect_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    blackhat_v = cv2.morphologyEx(masked_gray, cv2.MORPH_BLACKHAT, kernel_rect_v)

    # 合并水平+垂直方向的裂纹
    blackhat = cv2.addWeighted(blackhat_h, 0.5, blackhat_v, 0.5, 0)

    # 阈值化提取显著裂纹
    _, crack_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    crack_mask = cv2.bitwise_and(crack_mask, crack_mask, mask=mask)

    # 计算裂纹占比
    crack_pixels = (crack_mask > 0).sum()
    tongue_pixels = (mask > 0).sum()
    crack_ratio = crack_pixels / max(tongue_pixels, 1)

    # 连通域分析，统计裂纹条数
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(crack_mask, connectivity=8)
    crack_lines = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 15:
            crack_lines += 1

    detected = crack_ratio > 0.015 or crack_lines >= 3

    if detected:
        info = YOLO_CLASSES[7]
        return {"detected": True, "yolo_id": 7, "crack_score": round(crack_ratio, 4),
                "crack_lines": crack_lines, "description": info["tcm_meaning"]}

    return {"detected": False, "yolo_id": None, "crack_score": round(crack_ratio, 4),
            "crack_lines": crack_lines, "description": "未检测到明显裂纹"}


# ============================================================
# 4. 齿痕舌检测
# ============================================================

def detect_teeth_marks(mask: np.ndarray) -> dict:
    """
    检测舌体边缘的齿痕（牙齿压迫痕迹）。

    方法：比较实际轮廓与凸包，如果边缘有显著凹陷（波浪状）则为齿痕。

    参数:
        mask: 舌体二值掩膜

    返回:
        {"detected": bool, "yolo_id": 8/None, "indentation_score": float, "description": str}
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"detected": False, "yolo_id": None, "indentation_score": 0, "description": "无舌体轮廓"}

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 100:
        return {"detected": False, "yolo_id": None, "indentation_score": 0, "description": "舌体区域过小"}

    # 计算凸包
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    cnt_area = cv2.contourArea(cnt)

    if hull_area < 1:
        return {"detected": False, "yolo_id": None, "indentation_score": 0, "description": "无法计算凸包"}

    # 凹陷度：凸包面积与实际轮廓面积之比
    # 正常舌体接近1.0，齿痕舌有边缘凹陷，比值会更大
    indentation_ratio = hull_area / cnt_area

    # 进一步分析轮廓边缘的波纹数量
    # 用 approxPolyDP 简化轮廓，计算顶点数
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    num_vertices = len(approx)

    # 正常舌体轮廓顶点约 6-12 个，齿痕舌边缘波浪多，顶点数更多
    waviness = num_vertices / max(peri / 50, 1)  # 归一化波纹密度

    # 综合评分
    indentation_score = (indentation_ratio - 1.0) + waviness * 0.1

    detected = indentation_ratio > 1.15 or (indentation_ratio > 1.08 and num_vertices > 20)

    if detected:
        info = YOLO_CLASSES[8]
        return {"detected": True, "yolo_id": 8,
                "indentation_score": round(indentation_score, 3),
                "indentation_ratio": round(indentation_ratio, 3),
                "num_vertices": num_vertices,
                "description": info["tcm_meaning"]}

    return {"detected": False, "yolo_id": None,
            "indentation_score": round(indentation_score, 3),
            "indentation_ratio": round(indentation_ratio, 3),
            "num_vertices": num_vertices,
            "description": "边缘光滑，无明显齿痕"}


# ============================================================
# 5. 舌面脏腑分区凹凸分析
# ============================================================

def analyze_organ_regions(image_array: np.ndarray, mask: np.ndarray) -> dict:
    """
    将舌面按中医理论分为四区（心肺/脾胃/肾/肝胆），
    分析各区域的凹凸状态。

    中医舌面分区理论：
    - 舌尖（上 1/3）→ 心肺
    - 舌中（中 1/3）→ 脾胃
    - 舌根（下 1/3）→ 肾
    - 舌边（左右两侧）→ 肝胆

    凹凸判断依据：
    - 凹（正气虚）：该区域颜色偏暗/偏薄/纹理少
    - 凸（邪气实）：该区域颜色偏亮/偏厚/纹理多（如厚苔）

    参数:
        image_array: RGB 图像数组
        mask: 舌体掩膜

    返回:
        {"心肺": {"state": "凹"/"凸"/None, "yolo_id": 19/20/None, ...},
         "脾胃": {...}, "肾": {...}, "肝胆": {...}}
    """
    if mask.sum() == 0:
        return {organ: {"state": None, "yolo_id": None, "description": "无舌体区域"}
                for organ in ORGAN_REGIONS}

    h_img, w_img = mask.shape[:2]

    # 获取舌体区域的边界
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return {organ: {"state": None, "yolo_id": None, "description": "无舌体区域"}
                for organ in ORGAN_REGIONS}

    y_top, y_bottom = ys.min(), ys.max()
    x_left, x_right = xs.min(), xs.max()
    tongue_h = y_bottom - y_top
    tongue_w = x_right - x_left

    if tongue_h < 10 or tongue_w < 10:
        return {organ: {"state": None, "yolo_id": None, "description": "舌体区域过小"}
                for organ in ORGAN_REGIONS}

    # 计算整体舌体的平均亮度和纹理（作为基准）
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    tongue_pixels = gray[mask > 0]
    overall_brightness = float(np.mean(tongue_pixels)) if len(tongue_pixels) > 0 else 128

    # 计算整体纹理（Laplacian 方差）
    tongue_region_gray = cv2.bitwise_and(gray, gray, mask=mask)
    laplacian = cv2.Laplacian(tongue_region_gray, cv2.CV_64F)
    laplacian_masked = laplacian[mask > 0]
    overall_texture = float(np.std(laplacian_masked)) if len(laplacian_masked) > 0 else 0

    # 定义四区的坐标范围（基于舌体边界，归一化）
    y_third_1 = y_top + tongue_h // 3
    y_third_2 = y_top + tongue_h * 2 // 3
    x_third_1 = x_left + tongue_w // 3
    x_third_2 = x_left + tongue_w * 2 // 3

    regions = {
        "心肺": {
            "y_range": (y_top, y_third_1),
            "x_range": (x_third_1, x_third_2),
            "yolo_deficiency_id": 19,  # 心肺凹
            "yolo_excess_id": 20,      # 心肺凸
        },
        "脾胃": {
            "y_range": (y_third_1, y_third_2),
            "x_range": (x_third_1, x_third_2),
            "yolo_deficiency_id": 17,  # 脾胃凹
            "yolo_excess_id": 18,      # 脾胃凸
        },
        "肾": {
            "y_range": (y_third_2, y_bottom),
            "x_range": (x_third_1, x_third_2),
            "yolo_deficiency_id": 13,  # 肾凹
            "yolo_excess_id": 14,      # 肾凸
        },
        "肝胆": {
            "y_range": (y_third_1, y_third_2),
            "x_range_left": (x_left, x_third_1),
            "x_range_right": (x_third_2, x_right),
            "yolo_deficiency_id": 15,  # 肝胆凹
            "yolo_excess_id": 16,      # 肝胆凸
        },
    }

    result = {}
    for organ, region_def in regions.items():
        y1, y2 = region_def["y_range"]

        if "x_range" in region_def:
            x1, x2 = region_def["x_range"]
            region_mask = np.zeros_like(mask)
            region_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        else:
            # 肝胆：左右两侧
            x1_l, x2_l = region_def["x_range_left"]
            x1_r, x2_r = region_def["x_range_right"]
            region_mask = np.zeros_like(mask)
            region_mask[y1:y2, x1_l:x2_l] = mask[y1:y2, x1_l:x2_l]
            region_mask[y1:y2, x1_r:x2_r] = mask[y1:y2, x1_r:x2_r]

        region_pixels = (region_mask > 0).sum()

        if region_pixels < 20:
            result[organ] = {
                "state": None,
                "yolo_id": None,
                "brightness": 0,
                "texture": 0,
                "description": f"{organ}区域像素不足",
            }
            continue

        # 计算该区域的亮度和纹理
        region_gray = gray[region_mask > 0]
        region_brightness = float(np.mean(region_gray))

        region_laplacian = laplacian[region_mask > 0]
        region_texture = float(np.std(region_laplacian))

        # 与整体对比
        brightness_diff = region_brightness - overall_brightness
        texture_diff = region_texture - overall_texture

        # 凹凸判断：
        # 凹（正气虚）：区域偏暗 + 纹理偏少
        # 凸（邪气实）：区域偏亮 + 纹理偏多（如厚苔、隆起）
        score = brightness_diff * 0.4 + texture_diff * 0.6

        if score < -8:
            state = "凹"
            yolo_id = region_def["yolo_deficiency_id"]
            info = YOLO_CLASSES[yolo_id]
            desc = info["tcm_meaning"]
        elif score > 8:
            state = "凸"
            yolo_id = region_def["yolo_excess_id"]
            info = YOLO_CLASSES[yolo_id]
            desc = info["tcm_meaning"]
        else:
            state = None
            yolo_id = None
            desc = f"{organ}区域无明显凹凸异常"

        result[organ] = {
            "state": state,
            "yolo_id": yolo_id,
            "brightness": round(region_brightness, 1),
            "texture": round(region_texture, 1),
            "brightness_diff": round(brightness_diff, 1),
            "texture_diff": round(texture_diff, 1),
            "description": desc,
        }

    return result


# ============================================================
# 6. 综合分析入口
# ============================================================

def analyze_all_features(image_array: np.ndarray, mask: np.ndarray) -> dict:
    """
    对舌象进行完整的21类特征分析。

    参数:
        image_array: RGB 图像数组
        mask: 舌体二值掩膜

    返回:
        包含所有检测结果的字典：
        - shape: 舌体形态分析结果
        - red_spots: 红点舌检测结果
        - cracks: 裂纹舌检测结果
        - teeth_marks: 齿痕舌检测结果
        - organ_regions: 脏腑分区凹凸分析结果
        - detected_yolo_ids: 检测到的 YOLO 类别 ID 列表
    """
    shape_result = analyze_tongue_shape(mask, image_array)
    red_spots_result = detect_red_spots(image_array, mask)
    cracks_result = detect_cracks(image_array, mask)
    teeth_marks_result = detect_teeth_marks(mask)
    organ_result = analyze_organ_regions(image_array, mask)

    # 汇总检测到的 YOLO 类别
    detected_ids = []
    if shape_result["yolo_id"] is not None:
        detected_ids.append(shape_result["yolo_id"])
    if red_spots_result["yolo_id"] is not None:
        detected_ids.append(red_spots_result["yolo_id"])
    if cracks_result["yolo_id"] is not None:
        detected_ids.append(cracks_result["yolo_id"])
    if teeth_marks_result["yolo_id"] is not None:
        detected_ids.append(teeth_marks_result["yolo_id"])
    for organ_data in organ_result.values():
        if organ_data["yolo_id"] is not None:
            detected_ids.append(organ_data["yolo_id"])

    return {
        "shape": shape_result,
        "red_spots": red_spots_result,
        "cracks": cracks_result,
        "teeth_marks": teeth_marks_result,
        "organ_regions": organ_result,
        "detected_yolo_ids": detected_ids,
    }
