"""
模型推理模块
============
提供舌象分析的核心推理功能。

本模块支持两种推理模式：
1. 规则推理模式（默认）：基于 HSV 颜色特征的规则分类，无需训练，开箱即用
2. 深度学习模式：使用 MobileNetV2 迁移学习模型，精度更高，需先训练

在真实项目中，可参考以下技术路线提升精度：
- 使用 UNet 进行精确舌体分割
- 使用 MobileNetV2/EfficientNet 进行舌色分类
- 参考 ZhongJing-OMNI 数据集进行训练
"""

import os
from typing import Optional

import numpy as np

# PyTorch 为可选依赖：未安装时自动回退到规则推理模式
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    transforms = None
    models = None

from .segmentation import segment_tongue, analyze_tongue_body_color, analyze_coating_color
from .knowledge import (
    get_tongue_body_info,
    get_coating_info,
    get_constitution,
    get_health_advice,
    TONGUE_BODY_TYPES,
    TONGUE_COATING_TYPES,
)
from .utils import load_image, create_color_swatch
from .llm import generate_tcm_commentary, generate_fallback_commentary

# 模型保存路径
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TONGUE_BODY_MODEL_PATH = os.path.join(MODEL_DIR, "tongue_body_mobilenetv2.pth")
COATING_MODEL_PATH = os.path.join(MODEL_DIR, "coating_mobilenetv2.pth")

# 分类标签
TONGUE_BODY_LABELS = list(TONGUE_BODY_TYPES.keys())
COATING_LABELS = list(TONGUE_COATING_TYPES.keys())

# 图像预处理（用于深度学习模型，仅在 torch 可用时定义）
if TORCH_AVAILABLE:
    MODEL_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    MODEL_TRANSFORM = None


def build_mobilenetv2(num_classes: int):
    """
    构建 MobileNetV2 迁移学习模型。
    使用 ImageNet 预训练权重，替换最后一层分类器。
    需要 PyTorch 已安装。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 未安装，无法构建深度学习模型。请运行: pip install torch torchvision")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # 冻结特征提取层
    for param in model.features.parameters():
        param.requires_grad = False
    # 替换分类器
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_tongue_body_model():
    """加载舌质分类深度学习模型（如果存在且 PyTorch 可用）"""
    if not TORCH_AVAILABLE:
        return None
    if not os.path.exists(TONGUE_BODY_MODEL_PATH):
        return None
    try:
        model = build_mobilenetv2(num_classes=len(TONGUE_BODY_LABELS))
        model.load_state_dict(torch.load(TONGUE_BODY_MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        print(f"[警告] 加载舌质模型失败: {e}")
        return None


def load_coating_model():
    """加载舌苔分类深度学习模型（如果存在且 PyTorch 可用）"""
    if not TORCH_AVAILABLE:
        return None
    if not os.path.exists(COATING_MODEL_PATH):
        return None
    try:
        model = build_mobilenetv2(num_classes=len(COATING_LABELS))
        model.load_state_dict(torch.load(COATING_MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        print(f"[警告] 加载舌苔模型失败: {e}")
        return None


def predict_with_model(model, image_array: np.ndarray, labels: list) -> tuple:
    """
    使用深度学习模型进行预测。
    返回 (预测标签键, 置信度, 各类别概率字典)
    """
    from PIL import Image as PILImage
    img = PILImage.fromarray(image_array)
    input_tensor = MODEL_TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    label_key = labels[predicted_idx]
    prob_dict = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    return label_key, confidence, prob_dict


def analyze_tongue(
    image_input,
    use_ml_model: bool = True,
    agnes_api_key: Optional[str] = None,
    agnes_model: str = "agnes-2.5-pro",
    use_llm: bool = True,
    backend: str = "hsv",
) -> dict:
    """
    舌象分析主函数 - 完整推理流程。

    流程:
    1. 图像加载与预处理
    2. 舌体分割（由 backend 指定后端：HSV 或 U2-Net）
    3. 舌质颜色分析
    4. 舌苔颜色分析
    5. 体质判断
    6. 生成健康科普建议
    7. 调用 Agnes AI 大模型生成中医辨证评语（可选）

    参数:
        image_input: PIL.Image / numpy数组 / 文件路径
        use_ml_model: 是否尝试使用深度学习模型（如果已训练）
        agnes_api_key: Agnes AI API Key（可选，优先使用；未提供则尝试本地/环境变量）
        agnes_model: Agnes AI 模型名称，默认 agnes-2.5-pro
        use_llm: 是否调用 Agnes AI 生成中医评语（默认 True）
        backend: 分割后端选择（默认 "hsv"）：
            - "hsv": HSV 色彩阈值法，精度高不含嘴唇，但边缘可能欠分割
            - "u2net": U2-Net 深度学习分割，召回率高覆盖全舌体，但可能含嘴唇/面颊
              若 rembg 未安装或推理异常，自动回退到 HSV

    返回:
        dict 包含完整分析结果（含 ai_commentary 字段）
    """
    # 步骤1: 加载图像
    img = load_image(image_input)
    img_array = np.array(img)

    # 步骤2: 舌体分割（由 backend 指定后端）
    seg_result = segment_tongue(image_input, backend=backend)
    mask = seg_result["mask"]
    # 使用与 mask 同尺寸的 resize 后图像进行颜色分析，避免尺寸不匹配
    resized_array = seg_result.get("resized_array", img_array)

    # 步骤3 & 4: 颜色分析
    # 优先尝试深度学习模型
    body_result = None
    coating_result = None
    ml_available = False

    if use_ml_model and seg_result["success"]:
        body_model = load_tongue_body_model()
        coating_model = load_coating_model()

        if body_model is not None:
            try:
                tongue_region = seg_result["tongue_region"]
                body_key, body_conf, body_probs = predict_with_model(
                    body_model, tongue_region, TONGUE_BODY_LABELS
                )
                body_info = get_tongue_body_info(body_key)
                body_result = {
                    "body_key": body_key,
                    "name": body_info["name"],
                    "description": body_info["description"],
                    "tcm_meaning": body_info["tcm_meaning"],
                    "health_status": body_info["health_status"],
                    "confidence": body_conf,
                    "probabilities": body_probs,
                    "method": "deep_learning",
                }
                ml_available = True
            except Exception as e:
                print(f"[信息] 舌质深度学习推理失败，回退到规则模式: {e}")

        if coating_model is not None and body_result is not None:
            try:
                tongue_region = seg_result["tongue_region"]
                coating_key, coating_conf, coating_probs = predict_with_model(
                    coating_model, tongue_region, COATING_LABELS
                )
                coating_info = get_coating_info(coating_key)
                coating_result = {
                    "coating_key": coating_key,
                    "name": coating_info["name"],
                    "description": coating_info["description"],
                    "tcm_meaning": coating_info["tcm_meaning"],
                    "health_status": coating_info["health_status"],
                    "confidence": coating_conf,
                    "probabilities": coating_probs,
                    "method": "deep_learning",
                }
            except Exception as e:
                print(f"[信息] 舌苔深度学习推理失败，回退到规则模式: {e}")

    # 回退到规则推理模式
    if body_result is None:
        body_result = analyze_tongue_body_color(resized_array, mask)
        body_result["method"] = "rule_based"
        body_result["confidence"] = None

    if coating_result is None:
        coating_result = analyze_coating_color(resized_array, mask)
        coating_result["method"] = "rule_based"
        coating_result["confidence"] = None

    # 步骤5: 体质判断
    constitution = get_constitution(body_result["body_key"], coating_result["coating_key"])

    # 步骤6: 生成健康建议
    advice = get_health_advice(body_result["body_key"], coating_result["coating_key"])

    # 获取检测到的舌体平均颜色
    from .utils import compute_mean_color
    mean_color = compute_mean_color(resized_array, mask)

    # 组装基础分析结果
    base_result = {
        "segmentation": seg_result,
        "tongue_body": body_result,
        "coating": coating_result,
        "constitution": constitution,
        "advice": advice,
        "mean_color": mean_color,
        "color_swatch_html": create_color_swatch(mean_color),
        "ml_model_used": ml_available,
        "image_size": img.size,
    }

    # 步骤7: 调用 Agnes AI 大模型生成中医辨证评语（可选）
    if use_llm:
        print("[TongueAI] 正在调用 Agnes AI 生成中医评语...")
        commentary_result = generate_tcm_commentary(
            base_result,
            user_api_key=agnes_api_key,
            model=agnes_model,
            temperature=0.3,
            max_tokens=8000,
        )

        if commentary_result["success"]:
            print(f"[TongueAI] Agnes AI 评语生成成功（模型: {commentary_result['model']}）")
            ai_commentary = commentary_result["comment"]
            ai_commentary_source = "agnes_ai"
        else:
            # API 调用失败，回退到基于知识库的规则评语
            print(f"[TongueAI] Agnes AI 调用失败: {commentary_result['error']}，使用规则回退")
            ai_commentary = generate_fallback_commentary(base_result)
            ai_commentary_source = "rule_fallback"
    else:
        # 未启用 LLM，使用规则回退
        ai_commentary = generate_fallback_commentary(base_result)
        ai_commentary_source = "rule_disabled"

    base_result["ai_commentary"] = ai_commentary
    base_result["ai_commentary_source"] = ai_commentary_source
    base_result["ai_commentary_success"] = (
        use_llm and commentary_result["success"] if use_llm else False
    )

    return base_result
