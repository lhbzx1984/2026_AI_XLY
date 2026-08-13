"""
舌象分析报告生成模块
====================
将分析结果生成美观的 HTML 报告，支持浏览器打印导出 PDF。

报告特点：
- 自包含 HTML（图片以 base64 内嵌，无外部依赖）
- 打印优化 CSS（A4 纸张适配，分页控制）
- 医疗专业风格设计
- 可直接通过浏览器"打印 → 另存为 PDF"导出
"""

import base64
import datetime
import io
import re
from typing import Optional

import numpy as np
from PIL import Image


def _markdown_to_html(md_text: str) -> str:
    """
    将简易 Markdown 文本转换为 HTML（不依赖外部库）。

    支持的语法：
    - ## 标题、### 副标题
    - **加粗**
    - - 无序列表
    - 普通段落（保留换行）

    用于将 Agnes AI 返回的 Markdown 评语渲染到报告 HTML 中。
    """
    if not md_text:
        return ""

    lines = md_text.split("\n")
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # 列表项结束检测
        if in_list and not stripped.startswith("- "):
            html_lines.append("</ul>")
            in_list = False

        # 二级标题
        if stripped.startswith("## "):
            html_lines.append(f'<h4 style="color:#1a1a2e;margin:16px 0 8px;">{stripped[3:]}</h4>')
        # 三级标题
        elif stripped.startswith("### "):
            html_lines.append(f'<h5 style="color:#333;margin:12px 0 6px;">{stripped[4:]}</h5>')
        # 无序列表
        elif stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul style=\"margin:6px 0 6px 20px;\">")
                in_list = True
            # 处理加粗
            item = stripped[2:]
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            html_lines.append(f"<li>{item}</li>")
        # 空行
        elif stripped == "":
            html_lines.append("")
        # 普通段落
        else:
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped)
            html_lines.append(f"<p style=\"margin:4px 0;\">{text}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _array_to_base64(image_array: np.ndarray, format: str = "PNG", max_size: int = 400) -> str:
    """
    将 numpy 图像数组转为 base64 编码的 data URL。

    参数:
        image_array: numpy 图像数组 (H, W, 3)
        format: 图片格式 (PNG / JPEG)
        max_size: 最大边长（像素），用于缩放以减小报告体积

    返回:
        data URL 字符串，可直接用于 <img src="...">
    """
    if image_array is None:
        return ""

    img = Image.fromarray(image_array.astype(np.uint8))

    # 等比缩放
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format=format, quality=85 if format == "JPEG" else None)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if format == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _rgb_to_hex(rgb: tuple) -> str:
    """RGB 元组转十六进制颜色字符串"""
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def generate_report_html(
    result: dict,
    original_image: Optional[np.ndarray] = None,
    for_print: bool = False,
) -> str:
    """
    生成完整的 HTML 分析报告。

    参数:
        result: analyze_tongue() 返回的分析结果字典
        original_image: 原始舌象图片（numpy 数组），用于报告中展示
        for_print: 是否为打印模式（True 时隐藏交互按钮）

    返回:
        完整的 HTML 字符串（自包含，无外部依赖）
    """
    seg = result["segmentation"]
    body = result["tongue_body"]
    coating = result["coating"]
    constitution = result["constitution"]
    advice = result["advice"]
    mean_color = result["mean_color"]
    ml_used = result["ml_model_used"]

    # AI 中医评语（来自 Agnes AI 大模型或规则回退）
    ai_commentary = result.get("ai_commentary", "")
    ai_commentary_source = result.get("ai_commentary_source", "rule_disabled")
    ai_commentary_success = result.get("ai_commentary_success", False)
    commentary_html = _markdown_to_html(ai_commentary)

    # 评语来源标签
    if ai_commentary_source == "agnes_ai":
        commentary_badge = "🤖 Agnes AI 大模型生成"
        commentary_badge_color = "#722ED1"
    elif ai_commentary_source == "rule_fallback":
        commentary_badge = "⚠️ 规则回退（API 不可用）"
        commentary_badge_color = "#FAAD14"
    else:
        commentary_badge = "📋 规则生成（未启用 AI）"
        commentary_badge_color = "#8C8C8C"

    # 时间戳
    now = datetime.datetime.now()
    date_str = now.strftime("%Y年%m月%d日 %H:%M")
    report_id = now.strftime("TCM%Y%m%d%H%M%S")

    # 图片 base64
    orig_img_b64 = _array_to_base64(original_image, "JPEG", 350) if original_image is not None else ""
    seg_img_b64 = _array_to_base64(seg.get("contour_image"), "PNG", 350) if seg.get("contour_image") is not None else ""

    # 颜色色块
    color_hex = _rgb_to_hex(mean_color)

    # 分析方法（结合分割后端 + 推理方式）
    _backend_labels = {
        "hsv": "HSV 色彩阈值分割",
        "hsv_fallback": "HSV 兜底（区域生长）分割",
        "u2net": "U2-Net 深度学习分割",
        "u2net_fallback": "U2-Net 兜底分割",
    }
    seg_backend = seg.get("backend", "hsv")
    seg_label = _backend_labels.get(seg_backend, seg_backend)
    inference_label = "深度学习模型" if ml_used else "规则推理（颜色特征）"
    method_label = f"{seg_label} + {inference_label}"

    # 置信度
    body_conf = f"{body['confidence']:.1%}" if body.get("confidence") is not None else "—"
    coating_conf = f"{coating['confidence']:.1%}" if coating.get("confidence") is not None else "—"

    # 分割覆盖率
    coverage = f"{seg['coverage']:.1%}" if seg.get("coverage") is not None else "—"
    seg_status = "成功" if seg.get("success") else "未成功"

    # 健康状态颜色映射
    health_colors = {
        "正常": "#52C41A",
        "基本正常": "#52C41A",
        "需关注": "#FAAD14",
        "偏寒": "#1890FF",
        "偏热": "#FF4D4F",
        "可能异常": "#FAAD14",
        "需就医": "#FF4D4F",
    }

    def _health_color(status: str) -> str:
        for key, color in health_colors.items():
            if key in status:
                return color
        return "#1890FF"

    body_health_color = _health_color(body.get("health_status", ""))
    coating_health_color = _health_color(coating.get("health_status", ""))

    # 操作按钮（仅非打印模式显示）：打印导出 PDF + 下载 HTML 报告
    print_button = "" if for_print else """
    <div class="report-actions no-print">
        <button class="btn-print" onclick="printTongueReport()" title="只打印报告内容（推荐用于导出 PDF）">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="6 9 6 2 18 2 18 6"/>
                <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/>
                <rect x="6" y="14" width="12" height="8"/>
            </svg>
            打印 / 导出 PDF
        </button>
        <button class="btn-download" onclick="downloadReportHTML()" title="下载独立 HTML 报告文件">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            下载报告
        </button>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TongueAI Pro · 舌象分析报告</title>
<style>
  /* ===== 基础重置 ===== */
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    background: #f0f2f5;
    color: #333;
    line-height: 1.7;
    padding: 20px;
  }}

  /* ===== 报告容器 ===== */
  .report {{
    max-width: 800px;
    margin: 0 auto;
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.08);
    overflow: hidden;
    padding: 48px;
  }}

  /* ===== 报告头部 ===== */
  .report-header {{
    text-align: center;
    padding-bottom: 24px;
    border-bottom: 3px solid;
    border-image: linear-gradient(90deg, #00B4D8, #FF6B35) 1;
    margin-bottom: 32px;
  }}

  .report-brand {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 12px;
  }}

  .report-brand-logo {{
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #00B4D8, #FF6B35);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: bold;
    font-size: 18px;
  }}

  .report-brand-name {{
    font-size: 22px;
    font-weight: 700;
    color: #1a1a2e;
  }}

  .report-title {{
    font-size: 28px;
    font-weight: 800;
    color: #1a1a2e;
    margin: 8px 0;
    background: linear-gradient(135deg, #00B4D8, #FF6B35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}

  .report-meta {{
    font-size: 13px;
    color: #888;
    display: flex;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
  }}

  .report-meta span {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}

  /* ===== 图片对比区 ===== */
  .image-section {{
    display: flex;
    gap: 24px;
    margin-bottom: 32px;
  }}

  .image-card {{
    flex: 1;
    text-align: center;
  }}

  .image-card-label {{
    font-size: 14px;
    font-weight: 600;
    color: #555;
    margin-bottom: 10px;
    padding: 4px 16px;
    background: #f5f7fa;
    border-radius: 20px;
    display: inline-block;
  }}

  .image-card img {{
    width: 100%;
    max-width: 320px;
    height: auto;
    border-radius: 8px;
    border: 2px solid #e8e8e8;
  }}

  /* ===== 分析卡片 ===== */
  .analysis-card {{
    background: #fafbfc;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
  }}

  .analysis-card::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
  }}

  .card-body::before {{ background: #00B4D8; }}
  .card-coating::before {{ background: #FF6B35; }}
  .card-constitution::before {{ background: #B388FF; }}
  .card-advice::before {{ background: #69F0AE; }}
  .card-commentary::before {{ background: #722ED1; }}

  /* ===== AI 评语卡片 ===== */
  .commentary-card {{
    background: linear-gradient(135deg, #f9f0ff 0%, #fff 100%);
    border: 1px solid #d3adf7;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
  }}

  .commentary-card::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: #722ED1;
  }}

  .commentary-content {{
    font-size: 14px;
    color: #333;
    line-height: 1.8;
  }}

  .commentary-content h4 {{
    font-size: 16px;
    font-weight: 700;
    color: #1a1a2e;
    margin: 16px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e8d5ff;
  }}

  .commentary-content h4:first-child {{
    margin-top: 0;
  }}

  .commentary-content ul {{
    margin: 6px 0 6px 20px;
    padding-left: 0;
  }}

  .commentary-content li {{
    margin: 4px 0;
  }}

  .commentary-content p {{
    margin: 4px 0;
  }}

  .card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 8px;
  }}

  .card-title {{
    font-size: 18px;
    font-weight: 700;
    color: #1a1a2e;
  }}

  .card-badge {{
    font-size: 12px;
    padding: 3px 12px;
    border-radius: 12px;
    font-weight: 600;
    color: #fff;
  }}

  .info-row {{
    display: flex;
    margin-bottom: 8px;
    font-size: 14px;
  }}

  .info-label {{
    width: 100px;
    flex-shrink: 0;
    color: #888;
    font-weight: 500;
  }}

  .info-value {{
    color: #333;
    flex: 1;
  }}

  .color-swatch {{
    display: inline-block;
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #ddd;
    vertical-align: middle;
    margin-right: 6px;
  }}

  .description-box {{
    background: #fff;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 14px;
    color: #555;
    margin-top: 12px;
    border: 1px solid #eee;
  }}

  /* ===== 健康建议 ===== */
  .advice-content {{
    font-size: 14px;
    color: #444;
    white-space: pre-wrap;
    line-height: 1.8;
  }}

  .advice-content strong {{
    color: #1a1a2e;
  }}

  /* ===== 免责声明 ===== */
  .disclaimer {{
    margin-top: 32px;
    padding: 20px;
    background: #FFFBE6;
    border: 1px solid #FFE58F;
    border-radius: 8px;
    font-size: 13px;
    color: #8C6D1F;
    line-height: 1.6;
  }}

  .disclaimer-title {{
    font-weight: 700;
    margin-bottom: 8px;
    color: #D48806;
  }}

  /* ===== 页脚 ===== */
  .report-footer {{
    text-align: center;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #eee;
    font-size: 12px;
    color: #aaa;
  }}

  /* ===== 打印按钮 ===== */
  .report-actions {{
    max-width: 800px;
    margin: 0 auto 20px;
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    flex-wrap: wrap;
  }}

  .btn-print {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    background: linear-gradient(135deg, #00B4D8, #0077B6);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(0, 180, 216, 0.3);
  }}

  .btn-print:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 180, 216, 0.4);
  }}

  .btn-download {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    background: linear-gradient(135deg, #FF6B35, #E85D04);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(255, 107, 53, 0.3);
  }}

  .btn-download:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4);
  }}

  /* ===== 打印样式 ===== */
  @media print {{
    body {{
      background: #fff;
      padding: 0;
      font-size: 12px;
    }}

    .report {{
      box-shadow: none;
      border-radius: 0;
      max-width: 100%;
      padding: 20mm 15mm;
    }}

    .no-print {{
      display: none !important;
    }}

    .analysis-card {{
      break-inside: avoid;
      page-break-inside: avoid;
    }}

    .report-header {{
      break-after: avoid;
    }}

    @page {{
      size: A4;
      margin: 0;
    }}
  }}
</style>
</head>
<body>

{print_button}

<div class="report">

  <!-- ===== 报告头部 ===== -->
  <div class="report-header">
    <div class="report-brand">
      <div class="report-brand-logo">T</div>
      <span class="report-brand-name">TongueAI · Pro</span>
    </div>
    <h1 class="report-title">中医 AI 舌象分析报告</h1>
    <div class="report-meta">
      <span>📅 {date_str}</span>
      <span>📋 报告编号: {report_id}</span>
      <span>🔬 分析方法: {method_label}</span>
    </div>
  </div>

  <!-- ===== 图片对比区 ===== -->
  <div class="image-section">
    <div class="image-card">
      <div class="image-card-label">📷 原始舌象</div>
      <img src="{orig_img_b64}" alt="原始舌象" />
    </div>
    <div class="image-card">
      <div class="image-card-label">🔬 舌体分割结果</div>
      <img src="{seg_img_b64}" alt="分割结果" />
    </div>
  </div>

  <!-- ===== 舌质分析 ===== -->
  <div class="analysis-card card-body">
    <div class="card-header">
      <div class="card-title">舌质分析</div>
      <div class="card-badge" style="background: {body_health_color};">{body.get('health_status', '—')}</div>
    </div>
    <div class="info-row">
      <div class="info-label">舌质类型</div>
      <div class="info-value"><strong>{body['name']}</strong></div>
    </div>
    <div class="info-row">
      <div class="info-label">中医含义</div>
      <div class="info-value">{body['tcm_meaning']}</div>
    </div>
    <div class="info-row">
      <div class="info-label">平均颜色</div>
      <div class="info-value">
        <span class="color-swatch" style="background: {color_hex};"></span>
        RGB{mean_color} ({color_hex})
      </div>
    </div>
    <div class="info-row">
      <div class="info-label">置信度</div>
      <div class="info-value">{body_conf}</div>
    </div>
    <div class="info-row">
      <div class="info-label">分割覆盖</div>
      <div class="info-value">{coverage}（{seg_status}）</div>
    </div>
    <div class="description-box">{body['description']}</div>
  </div>

  <!-- ===== 舌苔分析 ===== -->
  <div class="analysis-card card-coating">
    <div class="card-header">
      <div class="card-title">舌苔分析</div>
      <div class="card-badge" style="background: {coating_health_color};">{coating.get('health_status', '—')}</div>
    </div>
    <div class="info-row">
      <div class="info-label">舌苔类型</div>
      <div class="info-value"><strong>{coating['name']}</strong></div>
    </div>
    <div class="info-row">
      <div class="info-label">中医含义</div>
      <div class="info-value">{coating['tcm_meaning']}</div>
    </div>
    <div class="info-row">
      <div class="info-label">置信度</div>
      <div class="info-value">{coating_conf}</div>
    </div>
    <div class="description-box">{coating['description']}</div>
  </div>

  <!-- ===== 体质辨识 ===== -->
  <div class="analysis-card card-constitution">
    <div class="card-header">
      <div class="card-title">体质辨识</div>
      <div class="card-badge" style="background: #B388FF;">参考</div>
    </div>
    <div class="info-row">
      <div class="info-label">体质类型</div>
      <div class="info-value"><strong>{constitution['type']}</strong></div>
    </div>
    <div class="info-row">
      <div class="info-label">体质特征</div>
      <div class="info-value">{constitution['feature']}</div>
    </div>
    <div class="description-box">{constitution['description']}</div>
  </div>

  <!-- ===== 健康建议 ===== -->
  <div class="analysis-card card-advice">
    <div class="card-header">
      <div class="card-title">健康科普建议</div>
    </div>
    <div class="advice-content">{advice}</div>
  </div>

  <!-- ===== AI 中医辨证评语（Agnes AI 大模型生成） ===== -->
  <div class="commentary-card">
    <div class="card-header">
      <div class="card-title">🤖 AI 中医辨证评语</div>
      <div class="card-badge" style="background: {commentary_badge_color};">{commentary_badge}</div>
    </div>
    <div class="commentary-content">{commentary_html}</div>
  </div>

  <!-- ===== 免责声明 ===== -->
  <div class="disclaimer">
    <div class="disclaimer-title">⚠️ 重要免责声明</div>
    本报告由 AI 系统自动生成，仅作为健康教育演示用途，不构成医疗诊断或治疗建议。
    体质辨识为综合参考，实际体质需由专业中医师通过望闻问切四诊合参判断。
    如有健康不适，请及时就医。本项目为青少年科技训练营教育演示项目，非医疗器械。
  </div>

  <!-- ===== 页脚 ===== -->
  <div class="report-footer">
    TongueAI Pro · 中医AI舌诊辅助健康识别系统<br/>
    报告生成时间: {date_str} · 报告编号: {report_id}<br/>
    © 2026 青少年科技训练营 · Vibe Coding 项目
  </div>

</div>

</body>
</html>"""
    return html


def save_report_html(result: dict, original_image: Optional[np.ndarray], filepath: str) -> str:
    """
    将报告保存为独立 HTML 文件。

    参数:
        result: 分析结果字典
        original_image: 原始图片
        filepath: 保存路径

    返回:
        保存的文件路径
    """
    html = generate_report_html(result, original_image, for_print=False)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath
