"""
图像处理工具模块
================
提供舌象图像的预处理、颜色分析等通用工具函数。
"""

import numpy as np
from PIL import Image


def load_image(image_input) -> Image.Image:
    """
    加载图像，统一转为 RGB 模式。
    支持 PIL.Image、numpy 数组、文件路径等多种输入。
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        raise ValueError(f"不支持的图像输入类型: {type(image_input)}")
    return img


def resize_image(img: Image.Image, max_size: int = 512) -> Image.Image:
    """按比例缩放图像，最长边不超过 max_size，减少计算量"""
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def rgb_to_hsv(rgb: tuple) -> tuple:
    """将 RGB 颜色转换为 HSV 色彩空间"""
    r, g, b = [x / 255.0 for x in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    # Hue
    if diff == 0:
        h = 0
    elif mx == r:
        h = 60 * (((g - b) / diff) % 6)
    elif mx == g:
        h = 60 * ((b - r) / diff + 2)
    else:
        h = 60 * ((r - g) / diff + 4)

    # Saturation
    s = 0 if mx == 0 else (diff / mx) * 100

    # Value
    v = mx * 100

    return (h, s, v)


def compute_mean_color(image_array: np.ndarray, mask: np.ndarray = None) -> tuple:
    """
    计算图像区域的平均 RGB 颜色。
    可选 mask 指定感兴趣区域（如舌体分割结果）。
    """
    if mask is not None:
        # 仅计算 mask 区域内的像素
        masked = image_array[mask > 0]
        if len(masked) == 0:
            masked = image_array.reshape(-1, 3)
    else:
        masked = image_array.reshape(-1, 3)

    mean_rgb = np.mean(masked, axis=0)
    return tuple(int(x) for x in mean_rgb)


def compute_color_features(image_array: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    计算图像区域的颜色特征统计量。
    返回 RGB 均值、HSV 均值、颜色分布等特征。
    """
    if mask is not None:
        pixels = image_array[mask > 0].astype(np.float32)
    else:
        pixels = image_array.reshape(-1, 3).astype(np.float32)

    if len(pixels) == 0:
        pixels = image_array.reshape(-1, 3).astype(np.float32)

    # RGB 均值与标准差
    mean_rgb = np.mean(pixels, axis=0)
    std_rgb = np.std(pixels, axis=0)

    # 转换为 HSV 统计
    hsv_pixels = np.array([rgb_to_hsv(tuple(p)) for p in pixels[:500]])  # 采样500个点加速
    mean_hue = np.mean(hsv_pixels[:, 0])
    mean_sat = np.mean(hsv_pixels[:, 1])
    mean_val = np.mean(hsv_pixels[:, 2])

    # 红色程度（R 通道相对值）
    r, g, b = mean_rgb
    redness = r / (r + g + b + 1e-6)  # 红色占比
    paleness = 1.0 - redness  # 淡白程度（近似）

    return {
        "mean_rgb": tuple(int(x) for x in mean_rgb),
        "std_rgb": tuple(int(x) for x in std_rgb),
        "mean_hue": float(mean_hue),
        "mean_saturation": float(mean_sat),
        "mean_value": float(mean_val),
        "redness": float(redness),
        "paleness": float(paleness),
        "r_over_g": float(r / (g + 1e-6)),
        "r_over_b": float(r / (b + 1e-6)),
    }


def create_color_swatch(rgb: tuple, size: int = 60) -> str:
    """
    生成一个 HTML 颜色色块字符串，用于在界面中展示检测到的颜色。
    """
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
    return f'<div style="display:inline-block;width:{size}px;height:{size}px;background:{hex_color};border-radius:8px;border:1px solid #ccc;vertical-align:middle;"></div>'
