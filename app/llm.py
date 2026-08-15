"""
Agnes AI 大模型中医评语生成模块
================================
本模块封装 Agnes AI（兼容 OpenAI 风格接口）的调用逻辑，
用于根据舌象分析结果生成专业的中医辨证评语。

参考项目：https://github.com/FJCU-AI-APPLICATION/Tongue-Diagnosis
原项目使用 Gemini API，本项目替换为 Agnes AI 大模型。

核心流程：
1. 将舌象分析结果（舌质、舌苔、体质）渲染为预测块
2. 将预测块注入系统提示词的 {{PREDICTIONS}} 占位符
3. 调用 Agnes AI Chat Completions API 生成评语
4. 错误时返回带错误标记的回退文本，不阻塞主流程

重要声明：本模块生成的评语仅用于教育科普目的，不构成医疗诊断或治疗建议。
"""

import json
import os
from typing import Optional

import urllib.request
import urllib.error

from .atlas import YOLO_CLASSES

# ============================================================
# 配置常量
# ============================================================

# Agnes AI API 配置（兼容 OpenAI 风格接口）
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_CHAT_ENDPOINT = f"{AGNES_BASE_URL}/chat/completions"
AGNES_DEFAULT_MODEL = "agnes-2.5-pro"

# 系统提示词模板路径
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.md")

# 预测块占位符
PREDICTIONS_PLACEHOLDER = "{{PREDICTIONS}}"

# 问卷数据占位符
QUESTIONNAIRE_PLACEHOLDER = "{{QUESTIONNAIRE}}"

# 用户触发消息（固定，不随分析结果变化）
USER_TRIGGER = "请依据上述判读数据和问卷信息，按规则输出大众版中医舌诊报告（含疾病风险预测）。"

# 错误标记前缀
ERROR_STAMP = "⚠ AI评语生成失败："

# API Key 存储文件（用户自定义目录，避免提交到版本库）
API_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".agnes_api_key")

# 请求超时（秒）—— 推理模型(agnes-2.5-pro)需要较长思考时间
REQUEST_TIMEOUT = 180


# ============================================================
# API Key 管理
# ============================================================

def resolve_api_key(user_key: Optional[str] = None) -> str:
    """
    解析 Agnes AI API Key。

    优先级：
    1. 用户在 UI 中输入的 key（user_key 参数）
    2. 本地保存的 key 文件（.agnes_api_key）
    3. 环境变量 AGNES_API_KEY

    若均不可用，抛出 RuntimeError。
    """
    # 1. 用户临时输入
    if user_key and user_key.strip():
        return user_key.strip()

    # 2. 本地保存的 key 文件
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                saved = f.read().strip()
                if saved:
                    return saved
        except Exception:
            pass

    # 3. 环境变量
    env_key = os.environ.get("AGNES_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    raise RuntimeError("尚未设置 Agnes AI API Key，请在「设置」中输入或配置环境变量 AGNES_API_KEY")


def save_api_key(key: str) -> bool:
    """
    将 API Key 保存到本地文件（明文，仅本机使用）。

    返回是否保存成功。
    """
    try:
        key = key.strip()
        if not key:
            return False
        with open(API_KEY_FILE, "w", encoding="utf-8") as f:
            f.write(key)
        return True
    except Exception as e:
        print(f"[LLM] 保存 API Key 失败: {e}")
        return False


def load_api_key() -> Optional[str]:
    """从本地文件加载已保存的 API Key，不存在则返回 None。"""
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                return f.read().strip() or None
        except Exception:
            return None
    return None


def clear_api_key() -> bool:
    """清除本地保存的 API Key。"""
    try:
        if os.path.exists(API_KEY_FILE):
            os.remove(API_KEY_FILE)
        return True
    except Exception as e:
        print(f"[LLM] 清除 API Key 失败: {e}")
        return False


def get_api_key_fingerprint(key: Optional[str] = None) -> str:
    """
    返回 API Key 的指纹（前8位 + 长度），用于 UI 显示而不泄露完整 key。
    """
    try:
        resolved = resolve_api_key(key) if key else (load_api_key() or "")
    except RuntimeError:
        return "未设置"

    if not resolved:
        return "未设置"

    import hashlib
    sha = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
    return f"{sha[:8]}…（长度 {len(resolved)}）"


# ============================================================
# 系统提示词管理
# ============================================================

class PromptValidationError(ValueError):
    """系统提示词模板缺少 {{PREDICTIONS}} 占位符时抛出。"""


def load_system_prompt() -> str:
    """从文件加载系统提示词模板。"""
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise PromptValidationError(f"系统提示词文件不存在: {PROMPT_FILE}")


def validate_prompt(template: str) -> None:
    """校验提示词模板包含且仅包含一个 {{PREDICTIONS}} 占位符。"""
    count = template.count(PREDICTIONS_PLACEHOLDER)
    if count == 0:
        raise PromptValidationError(f"提示词必须包含 {PREDICTIONS_PLACEHOLDER} 标记，目前缺少。")
    if count > 1:
        raise PromptValidationError(f"提示词只能有一个 {PREDICTIONS_PLACEHOLDER} 标记，目前有 {count} 个。")


def render_system_prompt(template: str, predictions_block: str, questionnaire_block: str = "") -> str:
    """将预测块和问卷数据注入系统提示词模板。"""
    validate_prompt(template)
    rendered = template.replace(PREDICTIONS_PLACEHOLDER, predictions_block)
    if QUESTIONNAIRE_PLACEHOLDER in rendered:
        rendered = rendered.replace(QUESTIONNAIRE_PLACEHOLDER, questionnaire_block or "（用户未填写问卷）")
    return rendered


# ============================================================
# 预测块渲染
# ============================================================

def render_predictions_block(result: dict) -> str:
    """
    将舌象分析结果渲染为预测块文本，用于注入系统提示词。
    包含21类完整分类结果：舌质颜色、舌体形态、舌面特征、舌苔、脏腑分区凹凸。

    参数:
        result: analyze_tongue() 返回的分析结果字典

    返回:
        预测块字符串（每行一个判读项）
    """
    body = result.get("tongue_body", {})
    coating = result.get("coating", {})
    constitution = result.get("constitution", {})
    seg = result.get("segmentation", {})
    mean_color = result.get("mean_color", (0, 0, 0))
    features = result.get("features", {})
    yolo_labels = result.get("yolo_labels", [])

    lines = []

    # ===== 一、舌质颜色 =====
    body_name = body.get("name", "未知")
    body_meaning = body.get("tcm_meaning", "")
    body_conf = body.get("confidence")
    body_yolo_id = result.get("body_yolo_id", 0)
    if body_conf is not None:
        lines.append(f"- 舌质颜色：{body_name}（{body_meaning}，置信度 {body_conf:.2f}）[YOLO类{body_yolo_id}]")
    else:
        lines.append(f"- 舌质颜色：{body_name}（{body_meaning}）[YOLO类{body_yolo_id}]")

    # ===== 二、舌体形态 =====
    shape_data = features.get("shape", {})
    if shape_data.get("yolo_id") is not None:
        shape_name = YOLO_CLASSES.get(shape_data["yolo_id"], {}).get("name", "异常")
        lines.append(f"- 舌体形态：{shape_name}（{shape_data.get('description', '')}）"
                     f"[YOLO类{shape_data['yolo_id']}]")
    else:
        lines.append(f"- 舌体形态：正常（宽高比{shape_data.get('aspect_ratio', 'N/A')}，"
                     f"覆盖率{shape_data.get('coverage', 'N/A')}）")

    # ===== 三、舌面特征 =====
    red_spots = features.get("red_spots", {})
    cracks = features.get("cracks", {})
    teeth_marks = features.get("teeth_marks", {})

    feature_parts = []
    if red_spots.get("detected"):
        feature_parts.append(f"红点舌（检测到{red_spots.get('spot_count', 0)}个红点）[YOLO类6]")
    if cracks.get("detected"):
        feature_parts.append(f"裂纹舌（{cracks.get('crack_lines', 0)}条裂纹）[YOLO类7]")
    if teeth_marks.get("detected"):
        feature_parts.append(f"齿痕舌（凹陷度{teeth_marks.get('indentation_ratio', 'N/A')}）[YOLO类8]")

    if feature_parts:
        lines.append(f"- 舌面特征：{'；'.join(feature_parts)}")
    else:
        lines.append("- 舌面特征：未见明显异常（无红点、裂纹、齿痕）")

    # ===== 四、舌苔 =====
    coat_name = coating.get("name", "未知")
    coat_meaning = coating.get("tcm_meaning", "")
    coat_conf = coating.get("confidence")
    coat_yolo_id = result.get("coating_yolo_id", 1)
    if coat_conf is not None:
        lines.append(f"- 舌苔：{coat_name}（{coat_meaning}，置信度 {coat_conf:.2f}）[YOLO类{coat_yolo_id}]")
    else:
        lines.append(f"- 舌苔：{coat_name}（{coat_meaning}）[YOLO类{coat_yolo_id}]")

    # ===== 五、脏腑分区凹凸 =====
    organ_data = features.get("organ_regions", {})
    organ_parts = []
    for organ_name in ["心肺", "脾胃", "肾", "肝胆"]:
        od = organ_data.get(organ_name, {})
        state = od.get("state")
        if state:
            organ_parts.append(f"{organ_name}{state}[YOLO类{od.get('yolo_id', '?')}]")
        else:
            organ_parts.append(f"{organ_name}正常")

    lines.append(f"- 脏腑分区：{'；'.join(organ_parts)}")

    # ===== 六、体质 =====
    constit_type = constitution.get("type", "未知")
    constit_feature = constitution.get("feature", "")
    lines.append(f"- 体质初判：{constit_type}（{constit_feature}）")

    # ===== 七、检测到的 YOLO 类别总览 =====
    if yolo_labels:
        yolo_summary = "、".join(f"{yl['name']}(类{yl['id']})" for yl in yolo_labels)
        lines.append(f"- 21类检测结果：{yolo_summary}")

    # ===== 八、辅助信息 =====
    if mean_color:
        lines.append(f"- 检测舌体平均颜色：RGB({int(mean_color[0])}, {int(mean_color[1])}, {int(mean_color[2])})")

    coverage = seg.get("coverage")
    if coverage is not None:
        lines.append(f"- 舌体分割覆盖率：{coverage:.1%}")

    ml_used = result.get("ml_model_used", False)
    _backend_labels = {
        "hsv": "HSV 色彩阈值分割",
        "hsv_fallback": "HSV 兜底（区域生长）分割",
        "u2net": "U2-Net 深度学习分割",
        "u2net_fallback": "U2-Net 兜底分割",
    }
    seg_backend = seg.get("backend", "hsv")
    seg_label = _backend_labels.get(seg_backend, seg_backend)
    inference_label = "深度学习模型" if ml_used else "规则推理（颜色特征+形态分析）"
    lines.append(f"- 分析方法：{seg_label} + {inference_label}")

    return "\n".join(lines) if lines else "- （无可用判读数据）"


# ============================================================
# Agnes AI API 调用
# ============================================================

def _call_agnes_api(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str = AGNES_DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    top_p: float = 0.9,
) -> str:
    """
    调用 Agnes AI Chat Completions API（兼容 OpenAI 风格接口）。

    使用 urllib 避免引入额外依赖（requests 非本项目必需）。

    Agnes 的 agnes-2.5-pro / agnes-2.5-flash 是推理模型（类似 DeepSeek-R1），
    响应中 message.reasoning_content 是思考过程，message.content 是最终答案。
    当 max_tokens 不足时，模型可能只输出 reasoning_content 而 content 为空。
    因此默认 max_tokens=4000 给推理+生成留足空间。

    返回模型生成的文本内容（优先 content，回退 reasoning_content）。

    异常时抛出，由上层调用者捕获并生成回退文本。
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        AGNES_CHAT_ENDPOINT,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Agnes API 返回 HTTP {e.code}: {err_body[:300]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接 Agnes API: {e.reason}") from e

    # 解析 OpenAI 风格响应（推理模型：只取 content，绝不回退 reasoning_content）
    try:
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("Agnes API 返回的 choices 为空")
        message = choices[0].get("message", {})
        content = message.get("content", "") or ""
        finish_reason = choices[0].get("finish_reason", "unknown")

        # 只使用 content（推理模型的最终答案）
        # 注意：reasoning_content 是模型的内部思考过程，绝不能作为结果返回给用户
        # 如果 content 为空，说明 max_tokens 不足导致模型还在思考阶段就被截断，
        # 此时返回错误，由上层走规则回退（generate_fallback_commentary）
        if content.strip():
            return content.strip()

        raise RuntimeError(
            f"Agnes API 未生成最终回应（content 为空，finish_reason={finish_reason}）。"
            f"推理模型可能因 max_tokens 不足在思考阶段被截断，请增大 max_tokens 后重试。"
        )
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"解析 Agnes API 响应失败: {e}") from e


def generate_tcm_commentary(
    result: dict,
    user_api_key: Optional[str] = None,
    model: str = AGNES_DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 8000,
    questionnaire_text: str = "",
) -> dict:
    """
    根据舌象分析结果和问卷数据，调用 Agnes AI 生成中医辨证评语（含疾病风险预测）。

    参数:
        result: analyze_tongue() 返回的分析结果字典
        user_api_key: 用户在 UI 中输入的 API Key（可选，优先使用）
        model: Agnes AI 模型名称
        temperature: 生成温度（越低越确定）
        max_tokens: 最大生成 token 数（含推理模型思考过程，需足够大）
        questionnaire_text: 格式化的问卷文本（供大模型参考）

    返回:
        dict 包含:
        - comment: 生成的中医评语（Markdown 格式）
        - success: 是否成功调用
        - error: 失败时的错误信息（成功时为 None）
        - model: 使用的模型名称
        - predictions_block: 注入提示词的预测块文本
    """
    # 渲染预测块
    predictions_block = render_predictions_block(result)

    # 加载并渲染系统提示词（含问卷数据）
    try:
        template = load_system_prompt()
        system_prompt = render_system_prompt(template, predictions_block, questionnaire_text)
    except PromptValidationError as e:
        return {
            "comment": f"{ERROR_STAMP}{type(e).__name__}: {e}",
            "success": False,
            "error": str(e),
            "model": model,
            "predictions_block": predictions_block,
        }

    # 解析 API Key
    try:
        api_key = resolve_api_key(user_api_key)
    except RuntimeError as e:
        return {
            "comment": f"{ERROR_STAMP}{e}",
            "success": False,
            "error": str(e),
            "model": model,
            "predictions_block": predictions_block,
        }

    # 调用 Agnes AI API
    try:
        comment = _call_agnes_api(
            system_prompt=system_prompt,
            user_message=USER_TRIGGER,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "comment": comment,
            "success": True,
            "error": None,
            "model": model,
            "predictions_block": predictions_block,
        }
    except Exception as e:
        error_msg = str(e)
        # 超时错误时，自动降级到 agnes-2.5-flash（响应更快，约 30 秒）
        is_timeout = "timed out" in error_msg.lower() or "timeout" in type(e).__name__.lower()
        if is_timeout and model != "agnes-2.5-flash":
            print(f"[LLM] {model} 超时，自动降级到 agnes-2.5-flash 重试...")
            try:
                comment = _call_agnes_api(
                    system_prompt=system_prompt,
                    user_message=USER_TRIGGER,
                    api_key=api_key,
                    model="agnes-2.5-flash",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return {
                    "comment": comment,
                    "success": True,
                    "error": None,
                    "model": "agnes-2.5-flash",
                    "predictions_block": predictions_block,
                }
            except Exception as e2:
                return {
                    "comment": f"{ERROR_STAMP}{type(e2).__name__}: {e2}",
                    "success": False,
                    "error": str(e2),
                    "model": "agnes-2.5-flash",
                    "predictions_block": predictions_block,
                }
        return {
            "comment": f"{ERROR_STAMP}{type(e).__name__}: {e}",
            "success": False,
            "error": error_msg,
            "model": model,
            "predictions_block": predictions_block,
        }


def test_api_connection(api_key: str, model: str = AGNES_DEFAULT_MODEL) -> dict:
    """
    测试 API Key 是否有效（发送一个最小请求）。

    Agnes 的 agnes-2.5-pro / agnes-2.5-flash 是推理模型，会先输出 reasoning_content
    （思考过程），再输出 content（最终答案）。因此 max_tokens 需足够大（800），
    否则模型还在思考阶段就被截断（finish_reason=length），content 为空。

    返回:
        dict 包含:
        - valid: 是否有效
        - message: 测试结果消息
        - model: 测试使用的模型
    """
    try:
        resp = _call_agnes_api(
            system_prompt="你是一个测试助手。",
            user_message="请回复「连接成功」四个字。",
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_tokens=800,
        )
        return {
            "valid": True,
            "message": f"连接成功，模型 {model} 可用",
            "model": model,
            "response": resp,
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"连接失败: {e}",
            "model": model,
            "response": "",
        }


# ============================================================
# 回退评语生成（API 不可用时使用）
# ============================================================

def generate_fallback_commentary(result: dict) -> str:
    """
    当 Agnes AI 不可用时，基于知识库生成基础评语（非 AI 生成）。

    参数:
        result: analyze_tongue() 返回的分析结果字典

    返回:
        Markdown 格式的基础评语文本
    """
    body = result.get("tongue_body", {})
    coating = result.get("coating", {})
    constitution = result.get("constitution", {})
    advice = result.get("advice", "")

    body_name = body.get("name", "未知")
    body_meaning = body.get("tcm_meaning", "")
    coat_name = coating.get("name", "未知")
    coat_meaning = coating.get("tcm_meaning", "")
    constit_type = constitution.get("type", "未知")
    constit_feature = constitution.get("feature", "")
    constit_desc = constitution.get("description", "")

    return f"""## 主要中医体质
{constit_type}（{constit_feature}）

## 次要中医体质
基于舌质与舌苔综合判断，无明显次要倾向。

## 体质说明
- 舌质表现为「{body_name}」，中医含义为「{body_meaning}」。
- 舌苔表现为「{coat_name}」，中医含义为「{coat_meaning}」。
- 综合判断体质为「{constit_type}」：{constit_desc}

## 证素列表
依据舌质与舌苔对应表推导（详见知识库）。

## 健康建议
{advice}

## 警语
此为系统基于规则生成（未使用 AI 大模型），仅供参考。
若有疾病或疑问，应向专业中医师咨询。
"""
