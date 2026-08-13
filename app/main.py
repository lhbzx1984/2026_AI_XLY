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
    if not api_key or not api_key.strip():
        return "❌ 请先输入 API Key"
    model = model_name.strip() if model_name and model_name.strip() else AGNES_DEFAULT_MODEL
    print(f"[TongueAI] 正在测试 Agnes AI 连接（模型: {model}）...")
    result = test_api_connection(api_key.strip(), model=model)
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


def run_analysis(image, consent_checked, minor_consent_checked, agnes_api_key, agnes_model, use_llm, backend):
    """
    Gradio 回调函数：执行舌象分析并生成结果展示。

    参数:
        image: 用户上传/拍摄的舌象图片
        consent_checked: 是否勾选免责声明确认
        minor_consent_checked: 是否勾选未成年人监护人同意
        agnes_api_key: Agnes AI API Key（用户在设置中输入）
        agnes_model: Agnes AI 模型名称
        use_llm: 是否启用 AI 评语生成
        backend: 分割后端选择（"hsv" 或 "u2net"）
            - "hsv": HSV 色彩阈值法（精度高不含嘴唇，但边缘可能欠分割）
            - "u2net": U2-Net 深度学习分割（召回率高覆盖全舌体，但可能含嘴唇/面颊）

    返回:
        多个 Gradio 组件的更新值
    """
    # 检查必要确认
    if not consent_checked:
        return (
            None, "❌ 请先阅读并勾选免责声明确认框，才能进行舌象分析。",
            "", "", "", "", "", "",
        )

    if not minor_consent_checked:
        return (
            None, "❌ 未成年人使用须获得监护人同意，请勾选确认框。",
            "", "", "", "", "", "",
        )

    # 检查图片
    if image is None:
        return (
            None, "❌ 请先上传或拍摄一张舌象照片。",
            "", "", "", "", "", "",
        )

    # 若启用 AI 评语但未提供 Key，给出提示（仍可分析，评语走规则回退）
    effective_key = agnes_api_key.strip() if agnes_api_key else None
    if use_llm and not effective_key:
        print("[TongueAI] 提示：已启用 AI 评语但未提供 API Key，将使用规则回退")

    try:
        # 执行分析（含 Agnes AI 评语生成）
        result = analyze_tongue(
            image,
            use_ml_model=True,
            agnes_api_key=effective_key,
            agnes_model=agnes_model if agnes_model else AGNES_DEFAULT_MODEL,
            use_llm=use_llm,
            backend=backend,
        )

        # 存储到全局变量，供报告生成使用
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

        # 生成舌体分割结果图
        contour_img = seg["contour_image"]

        # 生成舌质分析结果
        method_tag = "🤖 深度学习模型" if ml_used else "📊 规则推理（颜色特征）"
        confidence_str = ""
        if body.get("confidence") is not None:
            confidence_str = f"\n\n**模型置信度**: {body['confidence']:.1%}"

        body_result = f"""### 舌质分析：{body['name']}

**中医含义**: {body['tcm_meaning']}
**健康状态**: {body['health_status']}
**分析方法**: {method_tag}{confidence_str}

**描述**: {body['description']}
**检测到的舌体平均颜色**: {color_swatch} RGB{mean_color}
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

        # AI 中医评语
        ai_commentary = result.get("ai_commentary", "")
        commentary_source = result.get("ai_commentary_source", "rule_disabled")

        if commentary_source == "agnes_ai":
            source_label = f"✅ 评语来源：Agnes AI 大模型（{agnes_model if agnes_model else AGNES_DEFAULT_MODEL}）"
        elif commentary_source == "rule_fallback":
            source_label = "⚠️ 评语来源：规则回退（Agnes API 不可用，请检查 API Key）"
        else:
            source_label = "📋 评语来源：规则生成（未启用 Agnes AI）"

        commentary_result = f"""### 🤖 AI 中医辨证评语

{source_label}

---

{ai_commentary}

---

⚠️ **重要提醒**: AI 评语由大语言模型生成，仅供参考，不构成医疗诊断或治疗建议。
如有健康问题，请务必咨询专业中医师。
"""

        # 分割信息
        seg_info = ""
        if seg["success"]:
            backend = seg.get("backend", "hsv")
            backend_label = {
                "hsv": "HSV 色彩阈值法",
                "u2net": "U2-Net 深度学习",
                "u2net_fallback": "U2-Net 不可用，已回退到 HSV",
                "u2net_unavailable": "U2-Net 不可用，已回退到 HSV",
                "u2net_session_failed": "U2-Net 会话失败，已回退到 HSV",
            }.get(backend, backend)
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
            "",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            None, f"❌ 分析过程出错: {str(e)}\n\n请尝试更换照片或检查拍摄条件。",
            "", "", "", "", "", "",
        )


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

                # 当前 Key 状态显示
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
                # 左侧：输入
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

                # 右侧：分割结果
                with gr.Column(scale=1):
                    gr.Markdown("### 🔬 舌体分割结果")
                    seg_output = gr.Image(
                        label="舌体识别标注",
                        type="numpy",
                        height=300,
                    )
                    seg_info = gr.Markdown("")

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

            # 报告 HTML 展示区（带滚动）
            report_html = gr.HTML(
                value='<div style="text-align:center;padding:60px 20px;color:#888;'
                    'background:#fafafa;border-radius:12px;margin:20px 0;">'
                    '<p style="font-size:18px;margin-bottom:8px;">📋 点击「生成分析报告」查看完整报告</p>'
                    '<p style="font-size:14px;">完成舌象分析后，可生成报告并使用报告内的按钮导出 PDF 或下载 HTML</p>'
                    '</div>',
            )

            # 文件下载组件（备用方式：服务端生成 HTML 文件供下载）
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
            analyze_btn.click(
                fn=run_analysis,
                inputs=[image_input, consent, minor_consent,
                        agnes_api_key_input, agnes_model_input, use_llm_checkbox,
                        seg_backend_radio],
                outputs=[
                    seg_output,          # 分割结果图
                    seg_info,            # 分割信息
                    body_output,         # 舌质分析
                    coating_output,      # 舌苔分析
                    constitution_output, # 体质辨识
                    advice_output,       # 健康建议
                    commentary_output,   # AI 中医评语
                    gr.Textbox(visible=False),  # placeholder
                ],
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

            # 报告生成按钮：在页面内展示 HTML 报告
            report_btn.click(
                fn=generate_report_cb,
                inputs=[],
                outputs=[report_html],
            )

            # 导出按钮：保存 HTML 文件供下载
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
    # （沙盒内 httpx 访问 0.0.0.0 会失败返回 False，但服务器实际已正常运行）
    import gradio.networking
    import gradio.blocks
    gradio.networking.url_ok = lambda url: True
    gradio.blocks.networking.url_ok = lambda url: True

    app = create_app()

    # 禁用队列：沙盒环境中 SSE 长连接不稳定，会导致前端拿不到分析结果
    # Gradio 4.44 在 get_config_file() 中硬编码了 enable_queue=True，
    # 必须 monkey-patch 该方法才能让前端改用同步 HTTP 而非 SSE
    import gradio.blocks as _gb
    _orig_get_config_file = _gb.Blocks.get_config_file
    def _patched_get_config_file(self):
        config = _orig_get_config_file(self)
        config["enable_queue"] = False
        return config
    _gb.Blocks.get_config_file = _patched_get_config_file

    if hasattr(app, "_queue"):
        app._queue.enabled = False
    app.enable_queue = False

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设为 True 可生成公网临时链接（72小时，需 Gradio 隧道服务可用）
        show_error=True,
        allowed_paths=[
            os.path.join(os.path.dirname(__file__), "static", "images"),
            tempfile.gettempdir(),  # 允许下载临时目录中的报告文件
        ],
    )
