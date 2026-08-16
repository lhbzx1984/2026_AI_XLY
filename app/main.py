"""
中医AI舌诊辅助健康识别系统 - Gradio Web 应用主入口
=====================================================

启动方式:
    uv run python -m app.main

或直接运行:
    uv run gradio app/main.py

应用特点:
- 拍照/上传舌象图片即可分析
- 自动舌体分割与颜色特征提取
- 舌质、舌苔、体质三维度分析
- 健康科普建议（非诊断）
- 完整免责声明与未成年人保护
- 本地推理，图像不上传云端

⚠️ 重要：本项目为教育演示用途，非医疗器械，不用于临床诊断。
"""

import os
import sys
import datetime
import tempfile

import gradio as gr
import numpy as np
from PIL import Image

# 确保可以导入 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import analyze_tongue
from app.report import generate_report_html, save_report_html
from app.knowledge import (
    DISCLAIMER,
    PHOTO_GUIDE,
    TONGUE_BODY_TYPES,
    TONGUE_COATING_TYPES,
    CONSTITUTION_MAP,
)
from app.landing import get_landing_html, get_landing_css, get_landing_js, get_diagnosis_header_html, _img
from app.llm import (
    save_api_key,
    load_api_key,
    clear_api_key,
    get_api_key_fingerprint,
    test_api_connection,
    AGNES_DEFAULT_MODEL,
)
from app.questionnaire import QUESTIONNAIRE
from app.atlas import YOLO_CLASSES


# ===== 全局状态：存储最近一次分析结果（用于报告生成）=====
_last_result = None
_last_image = None


# ============================================================
# 中医舌诊图谱 HTML 生成（医院标准参考）
# ============================================================
# 健康状态 -> 颜色映射
_STATUS_COLOR = {
    "正常": "#16a34a",
    "需关注": "#ea580c",
    "建议就医": "#dc2626",
}


def _atlas_card(info: dict, img_key: str) -> str:
    """
    生成单张舌诊图谱卡片的 HTML。

    参数:
        info: 知识库中该类型的字典（name/description/tcm_meaning/health_status/advice）
        img_key: 图片文件名（不含扩展名，对应 atlas 目录下的 jpg 文件）

    返回:
        单张卡片的 HTML 字符串
    """
    name = info.get("name", img_key)
    name_en = info.get("name_en", "")
    description = info.get("description", "")
    tcm_meaning = info.get("tcm_meaning", "")
    health_status = info.get("health_status", "")
    advice = info.get("advice", "")

    # base64 内嵌图片
    img_src = _img(f"atlas/{img_key}.jpg")
    img_html = (
        f'<img src="{img_src}" alt="{name}" loading="lazy" '
        f'style="width:100%;height:160px;object-fit:cover;display:block;" />'
        if img_src
        else '<div style="width:100%;height:160px;background:#f3f4f6;display:flex;'
        f'align-items:center;justify-content:center;color:#9ca3af;font-size:13px;">图片加载中</div>'
    )

    status_color = _STATUS_COLOR.get(health_status, "#6b7280")

    return f"""
    <div style="background:#fff;border-radius:12px;overflow:hidden;
         box-shadow:0 2px 8px rgba(0,0,0,0.08);border:1px solid #e5e7eb;
         display:flex;flex-direction:column;transition:transform 0.2s,box-shadow 0.2s;"
         onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 8px 24px rgba(0,0,0,0.15)';"
         onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 2px 8px rgba(0,0,0,0.08)';">
      {img_html}
      <div style="padding:14px;flex:1;display:flex;flex-direction:column;gap:6px;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
          <span style="font-size:17px;font-weight:700;color:#1f2937;">{name}</span>
          <span style="font-size:11px;font-weight:600;color:{status_color};
               background:{status_color}1a;padding:2px 10px;border-radius:12px;
               white-space:nowrap;">{health_status}</span>
        </div>
        <span style="font-size:11px;color:#9ca3af;font-style:italic;">{name_en}</span>
        <p style="font-size:13px;color:#4b5563;margin:0;line-height:1.5;">{description}</p>
        <p style="font-size:12px;color:#6b7280;margin:0;line-height:1.5;
           padding:6px 10px;background:#f9fafb;border-radius:6px;border-left:3px solid {status_color};">
           <strong>中医意义：</strong>{tcm_meaning}</p>
        <p style="font-size:12px;color:#6b7280;margin:0;line-height:1.5;">{advice}</p>
      </div>
    </div>
    """


def build_atlas_html() -> str:
    """
    生成完整的中医舌诊图谱 HTML（医院标准参考）。

    包含舌质图谱（5 种）和舌苔图谱（6 种），每种类型配有标准参考图片、
    名称、健康状态、描述、中医含义与建议。

    返回:
        完整的 HTML 字符串，用于 gr.HTML 组件展示
    """
    # 舌质图谱
    body_cards = "".join(
        _atlas_card(info, key) for key, info in TONGUE_BODY_TYPES.items()
    )

    # 舌苔图谱
    coating_cards = "".join(
        _atlas_card(info, key) for key, info in TONGUE_COATING_TYPES.items()
    )

    return f"""
    <div style="font-family:'Noto Sans CJK SC','Microsoft YaHei',sans-serif;">
      <div style="background:linear-gradient(135deg,#fef3c7,#fde68a);border-left:4px solid #d97706;
           padding:12px 16px;border-radius:8px;margin-bottom:20px;">
        <p style="margin:0;font-size:13px;color:#92400e;line-height:1.6;">
          <strong>🏥 医院标准参考图谱</strong>　以下舌诊图片为 AI 生成的教学示意图，
          用于辅助理解各舌象类型的视觉特征，不作为临床诊断依据。
         实际诊断请以专业中医师判读为准。
        </p>
      </div>

      <h3 style="color:#1f2937;font-size:18px;margin:0 0 14px 0;padding-bottom:8px;
         border-bottom:2px solid #dc2626;">👅 舌质图谱（望舌体 — 反映脏腑气血盛衰）</h3>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
           gap:16px;margin-bottom:28px;">
        {body_cards}
      </div>

      <h3 style="color:#1f2937;font-size:18px;margin:0 0 14px 0;padding-bottom:8px;
         border-bottom:2px solid #0891b2;">👅 舌苔图谱（望舌苔 — 反映胃气盛衰和病邪性质）</h3>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
           gap:16px;">
        {coating_cards}
      </div>
    </div>
    """


def generate_report_cb():
    """回调：生成 HTML 报告并返回给 gr.HTML 组件展示"""
    global _last_result, _last_image
    if _last_result is None:
        return (
            '<div style="text-align:center;padding:60px 20px;color:#888;'
            'background:#fafafa;border-radius:12px;margin:20px 0;">'
            '<p style="font-size:18px;margin-bottom:8px;">📋 暂无分析数据</p>'
            '<p style="font-size:14px;">请先上传舌象照片并完成分析，再生成报告。</p>'
            '</div>'
        )
    html = generate_report_html(_last_result, _last_image, for_print=False)
    return html


def export_report_file_cb():
    """回调：将报告保存为 HTML 文件并返回文件路径供下载。

    增强版：添加错误处理与日志，确保即使分析未完成也能给出明确反馈。
    """
    global _last_result, _last_image
    if _last_result is None:
        print("[TongueAI] export_report_file_cb: 暂无分析数据，无法导出报告")
        return None
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filepath = os.path.join(tempfile.gettempdir(), f"tongueai_report_{timestamp}.html")
        save_report_html(_last_result, _last_image, filepath)
        print(f"[TongueAI] export_report_file_cb: 报告已保存到 {filepath}")
        # 验证文件确实已生成
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"[TongueAI] 报告文件大小: {file_size} 字节")
            return filepath
        else:
            print(f"[TongueAI] 警告：报告文件未生成: {filepath}")
            return None
    except Exception as e:
        import traceback
        print(f"[TongueAI] 导出报告出错: {e}")
        traceback.print_exc()
        return None


# ============================================================
# Agnes AI API Key 管理回调
# ============================================================

def save_agnes_key_cb(api_key, model_name):
    """回调：保存 Agnes AI API Key 到本地文件。"""
    if not api_key or not api_key.strip():
        return "❌ 请先输入 API Key", get_api_key_fingerprint()
    ok = save_api_key(api_key)
    if ok:
        fingerprint = get_api_key_fingerprint()
        return f"✅ API Key 已保存到本地（指纹：`{fingerprint}`）", fingerprint
    else:
        return "❌ 保存失败，请检查文件权限", get_api_key_fingerprint()


def test_agnes_key_cb(api_key, model_name):
    """回调：测试 Agnes AI API Key 是否有效。"""
    # 优先使用输入框的 Key，为空则尝试从文件加载
    key = api_key.strip() if api_key and api_key.strip() else None
    if not key:
        try:
            from app.llm import resolve_api_key
            key = resolve_api_key()
        except RuntimeError:
            return "❌ 请先输入 API Key，或确认 `.agnes_api_key` 文件存在"
    model = model_name.strip() if model_name and model_name.strip() else AGNES_DEFAULT_MODEL
    print(f"[TongueAI] 正在测试 Agnes AI 连接（模型: {model}）...")
    result = test_api_connection(key, model=model)
    if result["valid"]:
        return f"✅ {result['message']}\n\n模型回复：`{result['response'][:100]}`"
    else:
        return f"❌ {result['message']}"


def clear_agnes_key_cb():
    """回调：清除本地保存的 Agnes AI API Key。"""
    ok = clear_api_key()
    if ok:
        return "✅ 已清除本地保存的 API Key", get_api_key_fingerprint()
    else:
        return "❌ 清除失败", get_api_key_fingerprint()


def run_tongue_analysis(image, consent_checked, minor_consent_checked, backend):
    """
    第一步：舌象分析（21类分类，不调用大模型）。
    分析完成后结果存储在全局变量，供第二步综合报告使用。

    参数:
        image: 用户上传/拍摄的舌象图片
        consent_checked: 是否勾选免责声明确认
        minor_consent_checked: 是否勾选未成年人监护人同意
        backend: 分割后端选择（"hsv" 或 "u2net"）

    返回:
        7个 Gradio 组件的更新值
    """
    # 检查必要确认
    if not consent_checked:
        return (
            None, "❌ 请先阅读并勾选免责声明确认框，才能进行舌象分析。",
            "", "", "", "", "",
        )

    if not minor_consent_checked:
        return (
            None, "❌ 未成年人使用须获得监护人同意，请勾选确认框。",
            "", "", "", "", "",
        )

    # 检查图片
    if image is None:
        return (
            None, "❌ 请先上传或拍摄一张舌象照片。",
            "", "", "", "", "",
        )

    try:
        # 执行舌象分析（不调用 LLM，仅做21类分类）
        result = analyze_tongue(
            image,
            use_ml_model=True,
            use_llm=False,
            backend=backend,
        )

        # 存储到全局变量，供综合报告使用
        global _last_result, _last_image
        _last_result = result
        _last_image = image

        seg = result["segmentation"]
        body = result["tongue_body"]
        coating = result["coating"]
        constitution = result["constitution"]
        advice = result["advice"]
        mean_color = result["mean_color"]
        color_swatch = result["color_swatch_html"]
        ml_used = result["ml_model_used"]
        features = result.get("features", {})
        yolo_labels = result.get("yolo_labels", [])

        # 生成舌体分割结果图
        contour_img = seg["contour_image"]

        # 分析方法标签
        _backend_labels = {
            "hsv": "HSV 色彩阈值",
            "hsv_fallback": "HSV 兜底（区域生长）",
            "u2net": "U2-Net 深度学习",
            "u2net_fallback": "U2-Net 兜底",
        }
        seg_backend = seg.get("backend", "hsv")
        seg_label = _backend_labels.get(seg_backend, seg_backend)
        inference_label = "🤖 深度学习模型" if ml_used else "📊 规则推理（颜色特征+形态分析）"
        method_tag = f"{seg_label}分割 + {inference_label}"
        confidence_str = ""
        if body.get("confidence") is not None:
            confidence_str = f"\n\n**模型置信度**: {body['confidence']:.1%}"

        # ===== 21类特征详情 =====
        shape_data = features.get("shape", {})
        red_spots = features.get("red_spots", {})
        cracks = features.get("cracks", {})
        teeth_marks = features.get("teeth_marks", {})
        organ_data = features.get("organ_regions", {})

        # 舌体形态
        shape_lines = []
        if shape_data.get("yolo_id") is not None:
            shape_name = YOLO_CLASSES.get(shape_data["yolo_id"], {}).get("name", "异常")
            shape_lines.append(f"- **舌体形态**: {shape_name}（{shape_data.get('description', '')}）")
        else:
            shape_lines.append(f"- **舌体形态**: 正常（宽高比 {shape_data.get('aspect_ratio', 'N/A')}，覆盖率 {shape_data.get('coverage', 'N/A')}）")

        # 舌面特征
        feature_lines = []
        if red_spots.get("detected"):
            feature_lines.append(f"红点舌（{red_spots.get('spot_count', 0)}个红点）")
        if cracks.get("detected"):
            feature_lines.append(f"裂纹舌（{cracks.get('crack_lines', 0)}条裂纹）")
        if teeth_marks.get("detected"):
            feature_lines.append(f"齿痕舌（凹陷度 {teeth_marks.get('indentation_ratio', 'N/A')}）")
        feature_str = "、".join(feature_lines) if feature_lines else "未见明显异常"
        shape_lines.append(f"- **舌面特征**: {feature_str}")

        # 脏腑分区
        organ_parts = []
        for organ_name in ["心肺", "脾胃", "肾", "肝胆"]:
            od = organ_data.get(organ_name, {})
            state = od.get("state")
            if state:
                organ_parts.append(f"**{organ_name}{state}**（{od.get('description', '')}）")
            else:
                organ_parts.append(f"{organ_name}正常")
        shape_lines.append(f"- **脏腑分区**: {'；'.join(organ_parts)}")

        # YOLO 类别总览
        if yolo_labels:
            yolo_summary = "、".join(f"{yl['name']}" for yl in yolo_labels)
            shape_lines.append(f"- **21类检测结果**: {yolo_summary}")

        features_detail = "\n".join(shape_lines)

        body_result = f"""### 舌质分析：{body['name']}

**中医含义**: {body['tcm_meaning']}
**健康状态**: {body['health_status']}
**分析方法**: {method_tag}{confidence_str}

**描述**: {body['description']}
**检测到的舌体平均颜色**: {color_swatch} RGB{mean_color}

---

#### 21类分类详情

{features_detail}
"""

        # 生成舌苔分析结果
        confidence_str_c = ""
        if coating.get("confidence") is not None:
            confidence_str_c = f"\n\n**模型置信度**: {coating['confidence']:.1%}"

        coating_result = f"""### 舌苔分析：{coating['name']}

**中医含义**: {coating['tcm_meaning']}
**健康状态**: {coating['health_status']}
**分析方法**: {method_tag}{confidence_str_c}

**描述**: {coating['description']}
"""

        # 体质判断
        constitution_result = f"""### 体质辨识：{constitution['type']}

**体质特征**: {constitution['feature']}
**常见表现**: {constitution['description']}

> ⚠️ 体质辨识为综合参考，实际体质需由专业中医师通过望闻问切四诊合参判断。
"""

        # 健康建议
        advice_result = f"""### 健康科普建议

{advice}

---

⚠️ **再次提醒**: 以上分析结果仅供参考，不构成医疗诊断。如有健康不适，请及时就医。
本项目为教育演示用途，所有建议均为通用健康科普，不针对具体疾病。
"""

        # AI 评语占位（第二步生成）
        commentary_result = """### 🤖 AI 中医辨证评语

⏳ **请先填写下方问卷，然后点击「📋 生成综合报告」按钮**，系统将结合舌象21类分析结果与问卷数据，通过大模型生成综合体质辨识、健康建议和疾病风险预测。

---

⚠️ **重要提醒**: AI 评语由大语言模型生成，仅供参考，不构成医疗诊断或治疗建议。
"""

        # 分割信息
        seg_info = ""
        if seg["success"]:
            backend_label = {
                "hsv": "HSV 色彩阈值法",
                "u2net": "U2-Net 深度学习",
                "u2net_fallback": "U2-Net 不可用，已回退到 HSV",
                "u2net_unavailable": "U2-Net 不可用，已回退到 HSV",
                "u2net_session_failed": "U2-Net 会话失败，已回退到 HSV",
            }.get(seg_backend, seg_backend)
            seg_info = f"✅ 舌体分割成功，舌体面积占图像 **{seg['coverage']:.1%}**\n\n**分割后端**: {backend_label}"
        else:
            seg_info = "⚠️ 未能清晰识别舌体区域，分析结果可能不准确。请参考拍摄指南重新拍摄。"

        return (
            contour_img,
            seg_info,
            body_result,
            coating_result,
            constitution_result,
            advice_result,
            commentary_result,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            None, f"❌ 分析过程出错: {str(e)}\n\n请尝试更换照片或检查拍摄条件。",
            "", "", "", "", "",
        )


def generate_comprehensive_report(agnes_api_key, agnes_model, *questionnaire_values):
    """
    第二步：结合舌象分析结果 + 问卷数据，调用大模型生成综合报告。
    使用第一步存储的 _last_result 作为舌象分析数据。

    参数:
        agnes_api_key: Agnes AI API Key
        agnes_model: Agnes AI 模型名称
        *questionnaire_values: 问卷答案（Gradio Radio 返回的选项文本）

    返回:
        AI 中医评语 Markdown 文本
    """
    global _last_result

    if _last_result is None:
        return "❌ 请先完成第一步舌象分析，再来生成综合报告。"

    # 打包问卷答案（Gradio Radio 返回选项文本，需转为索引）
    questionnaire_answers = {}
    for i, q in enumerate(QUESTIONNAIRE):
        if i < len(questionnaire_values) and questionnaire_values[i] is not None:
            val = questionnaire_values[i]
            # Radio 返回的是选项文本，转换为索引
            if isinstance(val, str) and val in q["options"]:
                questionnaire_answers[q["id"]] = q["options"].index(val)
            elif isinstance(val, int) and 0 <= val < len(q["options"]):
                questionnaire_answers[q["id"]] = val

    # 解析 API Key
    effective_key = agnes_api_key.strip() if agnes_api_key else None
    if not effective_key:
        # 尝试从文件加载
        try:
            from app.llm import resolve_api_key
            effective_key = resolve_api_key()
        except RuntimeError:
            return """### 🤖 AI 中医辨证评语

❌ **未配置 Agnes AI API Key**，无法生成大模型综合报告。

请先在「🔑 Agnes AI 设置」中输入 API Key，或确认 `.agnes_api_key` 文件存在。

---

⚠️ 舌象21类分析结果已在上方显示，您仍可参考。如需 AI 综合分析，请配置 API Key 后重试。
"""

    # 格式化问卷数据
    questionnaire_text = ""
    if questionnaire_answers:
        from app.questionnaire import format_questionnaire_for_llm
        questionnaire_text = format_questionnaire_for_llm(questionnaire_answers)
        print(f"[TongueAI] 已注入问卷数据（{len(questionnaire_answers)}题），大模型将综合舌象+问卷进行分析")
    else:
        print("[TongueAI] 未填写问卷，大模型将仅基于舌象分析生成报告")

    # 调用大模型生成综合报告
    try:
        from app.llm import generate_tcm_commentary
        model_name = agnes_model if agnes_model else AGNES_DEFAULT_MODEL
        print(f"[TongueAI] 正在调用 Agnes AI（{model_name}）生成综合报告...")

        commentary_result = generate_tcm_commentary(
            _last_result,
            user_api_key=effective_key,
            model=model_name,
            temperature=0.3,
            max_tokens=8000,
            questionnaire_text=questionnaire_text,
        )

        if commentary_result["success"]:
            ai_commentary = commentary_result["comment"]
            source_label = f"✅ 评语来源：Agnes AI 大模型（{model_name}）"
        else:
            error_msg = commentary_result.get("error", "")
            ai_commentary = commentary_result.get("comment", "生成失败")
            if "insufficient_user_quota" in str(error_msg) or "额度" in str(error_msg):
                source_label = "⚠️ 评语来源：规则回退（Agnes AI 账户余额不足，请充值后重试）"
            elif error_msg:
                source_label = f"⚠️ 评语来源：规则回退（{error_msg[:80]}）"
            else:
                source_label = "⚠️ 评语来源：规则回退（Agnes API 不可用）"

        # 回写到 _last_result，供 HTML 报告使用
        _last_result["ai_commentary"] = ai_commentary
        _last_result["ai_commentary_source"] = "agnes_ai" if commentary_result["success"] else "rule_fallback"
        _last_result["ai_commentary_error"] = commentary_result.get("error")
        _last_result["ai_commentary_success"] = commentary_result["success"]
        # 注入问卷数据，供报告展示
        if questionnaire_answers:
            _last_result["questionnaire_answers"] = questionnaire_answers

        # 问卷结合提示
        questionnaire_note = ""
        if questionnaire_answers:
            questionnaire_note = "\n\n📝 **已结合问卷数据综合分析**（基础症状 + 生活习惯 + 既往病史），含疾病风险预测。"

        # 创建智能解读会话
        chat_btn = ""
        try:
            from app.chat_assistant import create_session
            session_id = create_session(_last_result)
            chat_btn = (
                f'<div style="text-align:center; margin: 24px 0;">'
                f'<a href="/chat?session={session_id}" target="_blank" '
                f'style="display:inline-block; background:linear-gradient(135deg, #2A9D8F, #1E7268); '
                f'color:#fff; padding:14px 36px; border-radius:12px; '
                f'text-decoration:none; font-size:16px; font-weight:600; '
                f'box-shadow:0 4px 12px rgba(42,157,143,0.3); transition:all 0.2s;" '
                f'onmouseover="this.style.transform=\'translateY(-2px)\';this.style.boxShadow=\'0 6px 20px rgba(42,157,143,0.4)\'" '
                f'onmouseout="this.style.transform=\'translateY(0)\';this.style.boxShadow=\'0 4px 12px rgba(42,157,143,0.3)\'">'
                f'🤖 智能解读咨询（AI 对话助手）'
                f'</a>'
                f'<p style="font-size:13px; color:#888; margin-top:8px;">'
                f'看不懂报告？点击与 AI 医生对话，获取个性化解读建议'
                f'</p>'
                f'</div>'
            )
        except Exception as cs_err:
            print(f"[TongueAI] 创建对话会话失败: {cs_err}")

        commentary_md = f"""### 🤖 AI 中医辨证评语

{source_label}{questionnaire_note}

---

{ai_commentary}

---

⚠️ **重要提醒**: AI 评语由大语言模型生成，仅供参考，不构成医疗诊断或治疗建议。
疾病风险预测仅为健康提示，如有健康问题，请务必咨询专业中医师或西医就诊。
"""
        return commentary_md, chat_btn

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_md = f"""### 🤖 AI 中医辨证评语

❌ 生成综合报告时出错: {str(e)}

请检查 API Key 配置和网络连接后重试。
"""
        return error_md, ""


# ============================================================
# 构建 Gradio 界面
# ============================================================

# Gradio 6.0+ 将 theme 和 css 参数移到 launch() 方法
APP_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="stone",
)

APP_CSS = """
.disclaimer-box {
    background: #FFF7E6;
    border: 1px solid #FAAD14;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}
.guide-box {
    background: #F6FFED;
    border: 1px solid #52C41A;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}
.main-title {
    text-align: center;
    color: #B8551D;
}
""" + get_landing_css()


# 通过 head 参数注入页面切换 JavaScript（gr.HTML 中的 <script> 不会被执行）
APP_HEAD = f"<script>{get_landing_js()}</script>"


def create_app():
    """创建并配置 Gradio 应用（双页面架构：主页 + 诊断页）"""

    with gr.Blocks(
        title="TongueAI Pro · 中医AI舌诊辅助健康识别系统",
        theme=APP_THEME,
        css=APP_CSS,
        head=APP_HEAD,
    ) as app:

        # ===== 页面1: 商业化科技风主页 =====
        with gr.Column(elem_id="tc-page-home"):
            gr.HTML(get_landing_html())

        # ===== 页面2: 诊断功能页 =====
        with gr.Column(elem_id="tc-page-diag"):

            # 诊断页顶部导航栏（含品牌标识 + 返回首页按钮）
            gr.HTML(get_diagnosis_header_html())

            # 诊断页标题栏
            gr.HTML(
                '<div class="tc-diag-hero-bar">'
                '<h2>智能舌象分析</h2>'
                '<p>上传舌象照片，AI 自动分析舌质舌苔与体质</p>'
                '</div>'
            )

            # ===== 免责声明 =====
            with gr.Row():
                gr.HTML(
                    f'<div class="disclaimer-box"><strong>⚠️ 重要免责声明</strong><br/>'
                    f'{DISCLAIMER.replace(chr(10), "<br/>")}</div>'
                )

            # ===== 伦理确认 =====
            with gr.Row():
                consent = gr.Checkbox(
                    label="我已阅读并理解上述免责声明，确认本项目不用于医疗诊断",
                    value=False,
                )
            with gr.Row():
                minor_consent = gr.Checkbox(
                    label="我是成年人，或我已获得监护人同意使用本系统",
                    value=False,
                )

            gr.Markdown("---")

            # ===== 功能页签：舌象诊断 =====
            with gr.Tabs():

                # ===== Tab: 舌象诊断 =====
                with gr.Tab("👅 舌象诊断"):
                    # ===== 拍摄指南 =====
                    with gr.Accordion("📸 拍摄指南（点击展开）", open=False):
                        guide_html = "<ol>"
                        for tip in PHOTO_GUIDE:
                            guide_html += f"<li>{tip}</li>"
                        guide_html += "</ol>"
                        gr.HTML(f'<div class="guide-box">{guide_html}</div>')

                    # ===== Agnes AI 大模型设置 =====
                    with gr.Accordion("🤖 Agnes AI 大模型设置（生成中医评语，点击展开）", open=False):
                        gr.HTML(
                            '<div style="background:#F9F0FF;border:1px solid #D3ADF7;border-radius:8px;'
                            'padding:12px 16px;margin:8px 0 16px;font-size:13px;color:#531DAB;">'
                            '<strong>💡 说明：</strong>配置 Agnes AI API Key 后，系统将调用大模型（替代原 Gemini API）'
                            '生成专业的中医辨证评语。未配置或未启用时，将使用基于知识库的规则评语作为回退。<br/>'
                            '<strong>获取 API Key：</strong>登录 <a href="https://platform.agnes-ai.com/" target="_blank">'
                            'Agnes AI 控制台</a>，进入 API Key 管理页面创建。'
                            '</div>'
                        )

                        _current_fingerprint = get_api_key_fingerprint()
                        agnes_key_status = gr.Markdown(
                            f"**当前 API Key 状态**：`{_current_fingerprint}`"
                        )

                        with gr.Row():
                            agnes_api_key_input = gr.Textbox(
                                label="Agnes AI API Key",
                                placeholder="在此粘贴你的 Agnes AI API Key（sk-... 或自定义格式）",
                                type="password",
                                lines=1,
                                scale=3,
                            )

                        with gr.Row():
                            save_key_btn = gr.Button("💾 保存 Key 到本地", variant="secondary", size="sm")
                            test_key_btn = gr.Button("🔌 测试连接", variant="secondary", size="sm")
                            clear_key_btn = gr.Button("🗑️ 清除已保存 Key", variant="stop", size="sm")

                        with gr.Row():
                            agnes_model_input = gr.Textbox(
                                label="模型名称",
                                value=AGNES_DEFAULT_MODEL,
                                placeholder="agnes-2.5-pro",
                                lines=1,
                                scale=2,
                            )
                            use_llm_checkbox = gr.Checkbox(
                                label="启用 Agnes AI 生成中医评语",
                                value=True,
                                scale=1,
                            )

                        key_test_result = gr.Markdown("")

                    # ===== 主交互区 =====
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📷 上传或拍摄舌象照片")
                            image_input = gr.Image(
                                label="舌象照片",
                                type="numpy",
                                sources=["upload", "webcam"],
                                height=300,
                            )
                            seg_backend_radio = gr.Radio(
                                label="分割后端（选择舌体识别算法）",
                                choices=[("HSV 色彩阈值法", "hsv"), ("U2-Net 深度学习", "u2net")],
                                value="hsv",
                                info="HSV：精度高不含嘴唇，但边缘可能欠分割 | "
                                     "U2-Net：召回率高覆盖全舌体，但可能含嘴唇/面颊。"
                                     "若 U2-Net 不可用（rembg 未安装）将自动回退到 HSV。",
                            )
                            analyze_btn = gr.Button(
                                "🔍 开始舌象分析",
                                variant="primary",
                                size="lg",
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("### 🔬 舌体分割结果")
                            seg_output = gr.Image(
                                label="舌体识别标注",
                                type="numpy",
                                height=300,
                            )
                            seg_info = gr.Markdown("")

                    # ===== 配套问卷（分析完成后填写，用于综合报告）=====
                    with gr.Accordion("📝 健康问卷（分析完成后填写，用于生成综合报告）", open=False):
                        gr.HTML(
                            '<div style="background:#FFF7E6;border:1px solid #FAAD14;border-radius:8px;'
                            'padding:12px 16px;margin:8px 0 16px;font-size:13px;color:#92400E;">'
                            '<strong>💡 说明：</strong>先完成上方舌象分析，再填写以下问卷。'
                            '填写后点击「📋 生成综合报告」，系统将结合舌象21类分析与问卷数据，'
                            '通过大模型综合给出体质辨识、健康建议和疾病风险预测。问卷数据仅在本次分析中使用，不会保存。'
                            '</div>'
                        )
                        questionnaire_components = []
                        current_section = ""
                        for q in QUESTIONNAIRE:
                            if q["section"] != current_section:
                                current_section = q["section"]
                                gr.Markdown(f"#### {current_section}")
                            radio = gr.Radio(
                                label=q["question"],
                                choices=q["options"],
                                value=None,
                            )
                            questionnaire_components.append(radio)

                        generate_report_btn = gr.Button(
                            "📋 生成综合报告",
                            variant="primary",
                            size="lg",
                        )

                    gr.Markdown("---")

                    # ===== 分析结果区 =====
                    gr.Markdown("## 📋 分析结果")

                    with gr.Row():
                        with gr.Column():
                            body_output = gr.Markdown("等待分析...")
                        with gr.Column():
                            coating_output = gr.Markdown("等待分析...")

                    with gr.Row():
                        with gr.Column():
                            constitution_output = gr.Markdown("")

                    gr.Markdown("---")

                    with gr.Row():
                        advice_output = gr.Markdown("")

                    gr.Markdown("---")

                    # ===== AI 中医评语区 =====
                    with gr.Row():
                        commentary_output = gr.Markdown("")

                    # ===== 报告生成与导出区 =====
                    gr.Markdown("---")
                    gr.Markdown("## 📄 分析报告")
                    gr.Markdown(
                        '<div style="background:#E6F7FF;border:1px solid #91D5FF;border-radius:8px;'
                        'padding:12px 16px;margin:8px 0 16px;font-size:13px;color:#003A8C;">'
                        '<strong>💡 使用提示：</strong>点击「生成分析报告」后，报告内将显示'
                        '<strong style="color:#00B4D8;">「打印 / 导出 PDF」</strong>和'
                        '<strong style="color:#FF6B35;">「下载报告」</strong>两个按钮。'
                        '前者通过浏览器打印对话框导出纯报告 PDF（不含网页其他内容），'
                        '后者下载独立 HTML 报告文件。'
                        '</div>'
                    )

                    with gr.Row():
                        report_btn = gr.Button(
                            "📋 生成分析报告",
                            variant="primary",
                            size="lg",
                        )
                        export_btn = gr.Button(
                            "💾 下载报告 HTML 文件（备用）",
                            variant="secondary",
                            size="lg",
                        )

                    report_html = gr.HTML(
                        value='<div style="text-align:center;padding:60px 20px;color:#888;'
                            'background:#fafafa;border-radius:12px;margin:20px 0;">'
                            '<p style="font-size:18px;margin-bottom:8px;">📋 点击「生成分析报告」查看完整报告</p>'
                            '<p style="font-size:14px;">完成舌象分析后，可生成报告并使用报告内的按钮导出 PDF 或下载 HTML</p>'
                            '</div>',
                    )

                    # 智能解读咨询按钮（生成综合报告后可用）
                    chat_btn_html = gr.HTML(
                        value='',
                        visible=True,
                    )

                    report_file = gr.File(
                        label="备用下载区（点击上方「下载报告 HTML 文件」按钮后，文件将显示在此处）",
                        visible=True,
                        interactive=False,
                    )

                    # ===== 知识库区 =====
                    gr.Markdown("---")
                    gr.Markdown("## 📚 中医舌诊知识库")
                    gr.Markdown(
                        "> 下方为 **医院标准参考图谱**，每种舌象类型配有示意图与中医释义，"
                        "可对照识别自身舌象特征。"
                    )
                    gr.HTML(build_atlas_html())

                    with gr.Accordion("体质类型对照表", open=False):
                        constit_table = "| 舌质 | 舌苔 | 体质类型 | 特征 |\n|------|------|----------|------|\n"
                        for (bk, ck), cinfo in CONSTITUTION_MAP.items():
                            b_name = TONGUE_BODY_TYPES.get(bk, {}).get("name", bk)
                            c_name = TONGUE_COATING_TYPES.get(ck, {}).get("name", ck)
                            constit_table += f"| {b_name} | {c_name} | {cinfo['type']} | {cinfo['feature']} |\n"
                        gr.Markdown(constit_table)

                    gr.Markdown("---")
                    gr.Markdown(
                        "### 📖 项目说明\n"
                        "本项目由青少年科技训练营开发，采用 **Vibe Coding** 范式构建。\n\n"
                        "**技术栈**: Python + Gradio + PyTorch (MobileNetV2) + OpenCV + Agnes AI 大模型\n\n"
                        "**分析方法**: 基于HSV颜色空间的舌体分割 + 颜色特征分析 + 中医知识映射 + Agnes AI 辨证评语\n\n"
                        "**AI 评语**: 通过 Agnes AI 大模型（兼容 OpenAI 接口）生成中医辨证评语，替代原 Gemini API\n\n"
                        "**伦理合规**: 非辅助决策类教育软件 · 数据本地处理 · 未成年人保护\n\n"
                        "---\n"
                        "⚠️ 本项目为教育演示用途，非医疗器械，不用于临床诊断。"
                    )

                # ===== 事件绑定 =====
                # 第一步：舌象分析（21类分类，不调用大模型）
                analyze_btn.click(
                    fn=run_tongue_analysis,
                    inputs=[image_input, consent, minor_consent,
                            seg_backend_radio],
                    outputs=[
                        seg_output,          # 分割结果图
                        seg_info,            # 分割信息
                        body_output,         # 舌质分析
                        coating_output,      # 舌苔分析
                        constitution_output, # 体质辨识
                        advice_output,       # 健康建议
                        commentary_output,   # AI 评语占位
                    ],
                )

                # 第二步：生成综合报告（大模型 + 问卷）
                generate_report_btn.click(
                    fn=generate_comprehensive_report,
                    inputs=[agnes_api_key_input, agnes_model_input]
                           + questionnaire_components,
                    outputs=[commentary_output, chat_btn_html],
                )

                # Agnes AI API Key 管理按钮
                save_key_btn.click(
                    fn=save_agnes_key_cb,
                    inputs=[agnes_api_key_input, agnes_model_input],
                    outputs=[key_test_result, agnes_key_status],
                )

                test_key_btn.click(
                    fn=test_agnes_key_cb,
                    inputs=[agnes_api_key_input, agnes_model_input],
                    outputs=[key_test_result],
                )

                clear_key_btn.click(
                    fn=clear_agnes_key_cb,
                    inputs=[],
                    outputs=[key_test_result, agnes_key_status],
                )

                # 报告生成按钮
                report_btn.click(
                    fn=generate_report_cb,
                    inputs=[],
                    outputs=[report_html],
                )

                export_btn.click(
                    fn=export_report_file_cb,
                    inputs=[],
                    outputs=[report_file],
                )

    return app


# ============================================================
# 启动应用
# ============================================================
if __name__ == "__main__":
    # 绕过 Gradio 4.44 在沙盒环境中的 localhost 连通性检查
    import gradio.networking
    import gradio.blocks
    gradio.networking.url_ok = lambda url: True
    gradio.blocks.networking.url_ok = lambda url: True

    # 禁用队列：沙盒环境中 SSE 长连接不稳定
    import gradio.blocks as _gb
    _orig_get_config_file = _gb.Blocks.get_config_file
    def _patched_get_config_file(self):
        config = _orig_get_config_file(self)
        config["enable_queue"] = False
        return config
    _gb.Blocks.get_config_file = _patched_get_config_file

    # 创建 FastAPI 应用，挂载 Gradio + 对话助手路由
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import gradio as gr
    import uvicorn

    fastapi_app = FastAPI()

    # ===== 对话助手路由 =====
    @fastapi_app.get("/chat", response_class=HTMLResponse)
    async def chat_page(session: str = ""):
        from app.chat_assistant import get_chat_page_html
        return get_chat_page_html(session)

    @fastapi_app.post("/chat/api/message")
    async def chat_message(data: dict):
        from app.chat_assistant import chat_with_agnes
        message = data.get("message", "")
        history = data.get("history", [])
        session_id = data.get("session_id", "")
        try:
            reply = chat_with_agnes(message, history, session_id)
            return JSONResponse({"reply": reply})
        except Exception as e:
            return JSONResponse({"error": f"对话服务出错: {str(e)[:100]}"})

    # ===== 创建并挂载 Gradio 主应用 =====
    gradio_app = create_app()

    if hasattr(gradio_app, "_queue"):
        gradio_app._queue.enabled = False
    gradio_app.enable_queue = False

    fastapi_app = gr.mount_gradio_app(
        fastapi_app,
        gradio_app,
        path="/",
        allowed_paths=[
            os.path.join(os.path.dirname(__file__), "static", "images"),
            tempfile.gettempdir(),
        ],
    )

    print("[TongueAI] 服务启动中...")
    print("[TongueAI] 主页面: http://127.0.0.1:7860/")
    print("[TongueAI] 对话助手: http://127.0.0.1:7860/chat")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
