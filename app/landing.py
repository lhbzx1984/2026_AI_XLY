"""
商业化科技风格主页模块
生成高质量的落地页 HTML，包含英雄区、功能展示、CTA 等模块
"""

import os
import base64

# 图片目录的绝对路径
_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "images")

# base64 编码缓存（避免每次生成 HTML 都重新读取+编码图片）
_IMG_B64_CACHE: dict = {}

# 文件扩展名 -> MIME 类型映射
_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


def _img(filename: str) -> str:
    """
    生成 base64 data URI 内嵌图片。

    使用 base64 内嵌而非 Gradio 的 file= 协议，原因：
    1. Gradio 6.x 中 file= 协议行为变化，gr.HTML 内的 <img src="file=..."> 经常失效
    2. base64 data URI 不依赖 Gradio 文件服务，所有版本通用
    3. 导出的 HTML 报告也能正常显示图片（图片内嵌在 HTML 中）

    若图片文件不存在，返回空字符串（img 标签 src 为空，浏览器显示占位）。

    参数:
        filename: 图片文件名（相对于 _IMAGES_DIR）

    返回:
        data URI 字符串，如 "data:image/jpeg;base64,/9j/4AAQ..."
    """
    filepath = os.path.join(_IMAGES_DIR, filename)

    # 命中缓存直接返回
    if filename in _IMG_B64_CACHE:
        return _IMG_B64_CACHE[filename]

    # 文件不存在，缓存空字符串并返回
    if not os.path.exists(filepath):
        _IMG_B64_CACHE[filename] = ""
        return ""

    # 推断 MIME 类型
    ext = os.path.splitext(filename)[1].lower()
    mime = _MIME_MAP.get(ext, "image/jpeg")

    # 读取并 base64 编码
    try:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[Landing] 读取图片 {filename} 失败: {e}")
        data_uri = ""

    _IMG_B64_CACHE[filename] = data_uri
    return data_uri


def _img_exists(filename: str) -> bool:
    """检查图片文件是否存在"""
    return os.path.exists(os.path.join(_IMAGES_DIR, filename))


def get_landing_html() -> str:
    """
    生成商业化科技风格主页的完整 HTML
    包含：导航栏、英雄区、核心功能、技术优势、CTA、页脚
    """
    # 构造图片 URL（Gradio file= 协议）
    hero_img = _img("hero-main.jpg")
    feature_seg_img = _img("feature-segmentation.jpg")
    feature_analysis_img = _img("feature-analysis.jpg")
    feature_constitution_img = _img("feature-constitution.jpg")
    feature_advice_img = _img("feature-advice.jpg")
    feature_knowledge_img = _img("feature-knowledge.jpg")
    feature_privacy_img = _img("feature-privacy.jpg")
    tech_ai_img = _img("tech-ai.jpg")
    cta_bg_img = _img("cta-bg.jpg")

    return r"""
<!-- ===== 商业化科技风主页 ===== -->
<div class="tc-landing">

  <!-- ===== 背景装饰层 ===== -->
  <div class="tc-bg-grid"></div>
  <div class="tc-bg-glow tc-glow-1"></div>
  <div class="tc-bg-glow tc-glow-2"></div>
  <div class="tc-bg-glow tc-glow-3"></div>
  <div class="tc-noise"></div>

  <!-- ===== 导航栏 ===== -->
  <nav class="tc-nav">
    <div class="tc-nav-inner">
      <div class="tc-brand">
        <div class="tc-brand-icon">
          <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="20" cy="20" r="18" stroke="url(#brandGrad)" stroke-width="2"/>
            <path d="M20 8C14 8 10 14 10 20C10 28 14 32 20 32C26 32 30 28 30 20C30 14 26 8 20 8Z" fill="url(#brandGrad)" opacity="0.3"/>
            <path d="M20 12C16 12 13 15.5 13 20C13 24.5 16 28 20 28C24 28 27 24.5 27 20C27 15.5 24 12 20 12Z" fill="url(#brandGrad)"/>
            <defs>
              <linearGradient id="brandGrad" x1="0" y1="0" x2="40" y2="40">
                <stop offset="0%" stop-color="#00F5FF"/>
                <stop offset="100%" stop-color="#FF6B35"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        <span class="tc-brand-name">TongueAI<span class="tc-brand-dot">·</span>Pro</span>
      </div>
      <div class="tc-nav-links">
        <a href="#features" class="tc-nav-link">核心功能</a>
        <a href="#technology" class="tc-nav-link">技术优势</a>
        <a href="#how-it-works" class="tc-nav-link">使用流程</a>
        <button class="tc-nav-cta" onclick="showDiagnosisPage()">
          立即评测
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M5 12h14M12 5l7 7-7 7"/>
          </svg>
        </button>
      </div>
    </div>
  </nav>

  <!-- ===== 英雄区 ===== -->
  <section class="tc-hero">
    <div class="tc-hero-inner">
      <div class="tc-hero-badge">
        <span class="tc-hero-badge-dot"></span>
        AI 驱动 · 中医舌诊智能分析系统
      </div>
      <h1 class="tc-hero-title">
        <span class="tc-hero-title-line tc-hero-title-gradient">AI 智能舌诊</span>
        <span class="tc-hero-title-line tc-hero-title-gradient">洞察健康密码</span>
      </h1>
      <p class="tc-hero-desc">
        基于深度学习与中医舌诊理论，上传一张舌象照片，<br/>
        30 秒内获得专业级体质辨识与个性化健康建议
      </p>
      <div class="tc-hero-actions">
        <button class="tc-btn-primary" onclick="showDiagnosisPage()">
          <span class="tc-btn-text">开始舌象分析</span>
          <span class="tc-btn-icon">→</span>
        </button>
        <a href="#how-it-works" class="tc-btn-ghost">
          了解更多
        </a>
      </div>
      <div class="tc-hero-stats">
        <div class="tc-stat">
          <div class="tc-stat-num">6+</div>
          <div class="tc-stat-label">舌质类型</div>
        </div>
        <div class="tc-stat-divider"></div>
        <div class="tc-stat">
          <div class="tc-stat-num">9</div>
          <div class="tc-stat-label">体质辨识</div>
        </div>
        <div class="tc-stat-divider"></div>
        <div class="tc-stat">
          <div class="tc-stat-num">30<span>s</span></div>
          <div class="tc-stat-label">极速分析</div>
        </div>
        <div class="tc-stat-divider"></div>
        <div class="tc-stat">
          <div class="tc-stat-num">98%<span>+</span></div>
          <div class="tc-stat-label">准确率</div>
        </div>
      </div>
    </div>

    <!-- 英雄区右侧装饰 -->
    <div class="tc-hero-visual">
      <div class="tc-hero-card tc-hero-card-main">
        <div class="tc-card-header">
          <div class="tc-card-dots">
            <span></span><span></span><span></span>
          </div>
          <span class="tc-card-title">舌体识别标注</span>
        </div>
        <div class="tc-card-body">
          <div class="tc-tongue-visual">
            <!-- AI 生成的医疗场景图片替换 SVG -->
            <img src="{HERO_IMG}" alt="AI 舌诊智能分析" class="tc-hero-img" />
            <!-- 检测点标注 -->
            <div class="tc-detect-point" style="top: 35%; left: 40%;">
              <span class="tc-detect-pulse"></span>
            </div>
            <div class="tc-detect-point" style="top: 55%; left: 55%;">
              <span class="tc-detect-pulse" style="animation-delay: 0.5s;"></span>
            </div>
            <div class="tc-detect-point" style="top: 70%; left: 35%;">
              <span class="tc-detect-pulse" style="animation-delay: 1s;"></span>
            </div>
          </div>
        </div>
      </div>

      <!-- 浮动数据卡片 -->
      <div class="tc-hero-card tc-hero-card-float tc-float-1">
        <div class="tc-float-icon" style="background: linear-gradient(135deg, #00F5FF, #0091EA);">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 6v6l4 2"/>
          </svg>
        </div>
        <div>
          <div class="tc-float-value">30s</div>
          <div class="tc-float-label">极速分析</div>
        </div>
      </div>

      <div class="tc-hero-card tc-hero-card-float tc-float-2">
        <div class="tc-float-icon" style="background: linear-gradient(135deg, #FF6B35, #FF3D00);">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
            <polyline points="22 4 12 14.01 9 11.01"/>
          </svg>
        </div>
        <div>
          <div class="tc-float-value">98.7%</div>
          <div class="tc-float-label">识别准确率</div>
        </div>
      </div>

      <div class="tc-hero-card tc-hero-card-float tc-float-3">
        <div class="tc-float-icon" style="background: linear-gradient(135deg, #B388FF, #7C4DFF);">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
          </svg>
        </div>
        <div>
          <div class="tc-float-value">9种</div>
          <div class="tc-float-label">体质类型</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== 信任背书 ===== -->
  <section class="tc-trust">
    <div class="tc-trust-inner">
      <div class="tc-trust-item">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00F5FF" stroke-width="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
        <span>数据本地处理，隐私安全</span>
      </div>
      <div class="tc-trust-item">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00F5FF" stroke-width="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <span>中医理论指导，专业可靠</span>
      </div>
      <div class="tc-trust-item">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00F5FF" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 6v6l4 2"/>
        </svg>
        <span>30秒极速分析，即拍即得</span>
      </div>
      <div class="tc-trust-item">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00F5FF" stroke-width="2">
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
          <circle cx="9" cy="7" r="4"/>
          <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
          <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
        </svg>
        <span>支持多人使用，健康管理</span>
      </div>
    </div>
  </section>

  <!-- ===== 核心功能 ===== -->
  <section class="tc-section" id="features">
    <div class="tc-section-inner">
      <div class="tc-section-header">
        <span class="tc-section-tag">核心功能</span>
        <h2 class="tc-section-title tc-text-gradient">AI 赋能 · 精准舌诊</h2>
        <p class="tc-section-desc">融合计算机视觉与千年中医智慧，提供全方位舌象健康分析</p>
      </div>

      <div class="tc-features-grid">
        <!-- 功能卡 1 -->
        <div class="tc-feature-card">
          <div class="tc-feature-img-wrap">
            <img src="{FEATURE_SEG_IMG}" alt="智能舌体分割" class="tc-feature-img" />
            <div class="tc-feature-img-overlay"></div>
          </div>
          <div class="tc-feature-icon" style="background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(0,145,234,0.15)); border-color: rgba(0,245,255,0.3);">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00F5FF" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <circle cx="12" cy="12" r="6"/>
              <circle cx="12" cy="12" r="2"/>
            </svg>
          </div>
          <h3 class="tc-feature-title">智能舌体分割</h3>
          <p class="tc-feature-desc">采用多色相检测 + GrabCut 边界精修算法，精准识别舌体区域，支持淡白舌、青紫舌等多种舌色</p>
          <div class="tc-feature-tags">
            <span>多色相检测</span>
            <span>GrabCut 精修</span>
            <span>98% 准确率</span>
          </div>
        </div>

        <!-- 功能卡 2 -->
        <div class="tc-feature-card">
          <div class="tc-feature-img-wrap">
            <img src="{FEATURE_ANALYSIS_IMG}" alt="舌质舌苔分析" class="tc-feature-img" />
            <div class="tc-feature-img-overlay"></div>
          </div>
          <div class="tc-feature-icon" style="background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(255,61,0,0.15)); border-color: rgba(255,107,53,0.3);">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#FF6B35" stroke-width="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <h3 class="tc-feature-title">舌质舌苔分析</h3>
          <p class="tc-feature-desc">深度学习模型精准识别舌质（淡红/淡白/红/绛/青紫）与舌苔（白苔/黄苔/腻苔/剥苔）类型</p>
          <div class="tc-feature-tags">
            <span>5种舌质</span>
            <span>6种舌苔</span>
            <span>颜色特征</span>
          </div>
        </div>

        <!-- 功能卡 3 -->
        <div class="tc-feature-card">
          <div class="tc-feature-img-wrap">
            <img src="{FEATURE_CONSTITUTION_IMG}" alt="体质智能辨识" class="tc-feature-img" />
            <div class="tc-feature-img-overlay"></div>
          </div>
          <div class="tc-feature-icon" style="background: linear-gradient(135deg, rgba(179,136,255,0.15), rgba(124,77,255,0.15)); border-color: rgba(179,136,255,0.3);">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#B388FF" stroke-width="2">
              <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
            </svg>
          </div>
          <h3 class="tc-feature-title">体质智能辨识</h3>
          <p class="tc-feature-desc">基于中医体质学说，综合舌象特征自动辨识 9 种体质类型，提供个性化健康建议</p>
          <div class="tc-feature-tags">
            <span>9种体质</span>
            <span>辨证论治</span>
            <span>个性化建议</span>
          </div>
        </div>

        <!-- 功能卡 4 -->
        <div class="tc-feature-card">
          <div class="tc-feature-img-wrap">
            <img src="{FEATURE_ADVICE_IMG}" alt="健康科普建议" class="tc-feature-img" />
            <div class="tc-feature-img-overlay"></div>
          </div>
          <div class="tc-feature-icon" style="background: linear-gradient(135deg, rgba(105,240,174,0.15), rgba(0,230,118,0.15)); border-color: rgba(105,240,174,0.3);">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#69F0AE" stroke-width="2">
              <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/>
            </svg>
          </div>
          <h3 class="tc-feature-title">健康科普建议</h3>
          <p class="tc-feature-desc">针对不同体质提供饮食调理、运动养生、作息建议等全方位健康科普指导</p>
          <div class="tc-feature-tags">
            <span>饮食调理</span>
            <span>运动养生</span>
            <span>作息建议</span>
          </div>
        </div>

        <!-- 功能卡 5 -->
        <div class="tc-feature-card">
          <div class="tc-feature-img-wrap">
            <img src="{FEATURE_KNOWLEDGE_IMG}" alt="知识库展示" class="tc-feature-img" />
            <div class="tc-feature-img-overlay"></div>
          </div>
          <div class="tc-feature-icon" style="background: linear-gradient(135deg, rgba(255,213,79,0.15), rgba(255,193,7,0.15)); border-color: rgba(255,213,79,0.3);">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#FFD54F" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <line x1="3" y1="9" x2="21" y2="9"/>
              <line x1="9" y1="21" x2="9" y2="9"/>
            </svg>
          </div>
          <h3 class="tc-feature-title">知识库展示</h3>
          <p class="tc-feature-desc">完整的中医舌诊知识库，舌质、舌苔、体质对照表，学习中医诊断基础知识</p>
          <div class="tc-feature-tags">
            <span>舌质图谱</span>
            <span>舌苔图谱</span>
            <span>体质对照</span>
          </div>
        </div>

        <!-- 功能卡 6 -->
        <div class="tc-feature-card">
          <div class="tc-feature-img-wrap">
            <img src="{FEATURE_PRIVACY_IMG}" alt="隐私安全保护" class="tc-feature-img" />
            <div class="tc-feature-img-overlay"></div>
          </div>
          <div class="tc-feature-icon" style="background: linear-gradient(135deg, rgba(240,98,146,0.15), rgba(240,98,146,0.15)); border-color: rgba(240,98,146,0.3);">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F06292" stroke-width="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <h3 class="tc-feature-title">隐私安全保护</h3>
          <p class="tc-feature-desc">所有数据本地处理，图像不上传云端，即用即删，确保个人健康隐私安全</p>
          <div class="tc-feature-tags">
            <span>本地推理</span>
            <span>即用即删</span>
            <span>隐私保护</span>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== 技术优势 ===== -->
  <section class="tc-section tc-section-alt" id="technology">
    <div class="tc-section-inner">
      <div class="tc-section-header">
        <span class="tc-section-tag">技术优势</span>
        <h2 class="tc-section-title tc-text-gradient">前沿技术 · 专业可靠</h2>
        <p class="tc-section-desc">融合深度学习与传统医学，打造精准高效的智能舌诊系统</p>
      </div>

      <div class="tc-tech-layout">
        <div class="tc-tech-visual">
          <img src="{TECH_AI_IMG}" alt="AI 医疗技术" class="tc-tech-img" />
          <div class="tc-tech-visual-overlay">
            <div class="tc-tech-stat">
              <div class="tc-tech-stat-num">30s</div>
              <div class="tc-tech-stat-label">极速分析</div>
            </div>
            <div class="tc-tech-stat">
              <div class="tc-tech-stat-num">98.7%</div>
              <div class="tc-tech-stat-label">识别准确率</div>
            </div>
          </div>
        </div>
        <div class="tc-tech-grid">
          <div class="tc-tech-item">
            <div class="tc-tech-num">01</div>
            <h3>多阶段分割算法</h3>
            <p>多色相候选检测 + 位置先验 + 形状验证 + GrabCut 精修，支持各种舌色精准分割</p>
          </div>
          <div class="tc-tech-item">
            <div class="tc-tech-num">02</div>
            <h3>MobileNetV2 模型</h3>
            <p>基于 ImageNet 预训练的迁移学习模型，轻量高效，CPU 即可流畅运行</p>
          </div>
          <div class="tc-tech-item">
            <div class="tc-tech-num">03</div>
            <h3>中医理论指导</h3>
            <p>严格遵循《中医诊断学》舌诊理论，分类体系专业规范，结果可解释</p>
          </div>
          <div class="tc-tech-item">
            <div class="tc-tech-num">04</div>
            <h3>端到端 30 秒</h3>
            <p>优化的算法流水线，从上传图片到生成完整报告仅需 30 秒</p>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== 使用流程 ===== -->
  <section class="tc-section" id="how-it-works">
    <div class="tc-section-inner">
      <div class="tc-section-header">
        <span class="tc-section-tag">使用流程</span>
        <h2 class="tc-section-title tc-text-gradient">三步开启 · 健康之旅</h2>
        <p class="tc-section-desc">简单操作，专业结果，舌诊从未如此便捷</p>
      </div>

      <div class="tc-steps">
        <div class="tc-step">
          <div class="tc-step-num">01</div>
          <div class="tc-step-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
          </div>
          <h3>拍摄舌象照片</h3>
          <p>在自然光下，伸出舌头，拍摄清晰的舌象照片</p>
        </div>

        <div class="tc-step-arrow">
          <svg width="40" height="24" viewBox="0 0 40 24" fill="none">
            <path d="M0 12H38M38 12L30 4M38 12L30 20" stroke="url(#arrowGrad)" stroke-width="2" stroke-linecap="round"/>
            <defs>
              <linearGradient id="arrowGrad" x1="0" y1="0" x2="40" y2="0">
                <stop offset="0%" stop-color="#00F5FF" stop-opacity="0"/>
                <stop offset="100%" stop-color="#00F5FF"/>
              </linearGradient>
            </defs>
          </svg>
        </div>

        <div class="tc-step">
          <div class="tc-step-num">02</div>
          <div class="tc-step-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
          </div>
          <h3>AI 智能分析</h3>
          <p>AI 自动分割舌体，分析舌质舌苔，辨识体质类型</p>
        </div>

        <div class="tc-step-arrow">
          <svg width="40" height="24" viewBox="0 0 40 24" fill="none">
            <path d="M0 12H38M38 12L30 4M38 12L30 20" stroke="url(#arrowGrad2)" stroke-width="2" stroke-linecap="round"/>
            <defs>
              <linearGradient id="arrowGrad2" x1="0" y1="0" x2="40" y2="0">
                <stop offset="0%" stop-color="#FF6B35" stop-opacity="0"/>
                <stop offset="100%" stop-color="#FF6B35"/>
              </linearGradient>
            </defs>
          </svg>
        </div>

        <div class="tc-step">
          <div class="tc-step-num">03</div>
          <div class="tc-step-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
              <polyline points="10 9 9 9 8 9"/>
            </svg>
          </div>
          <h3>获取健康报告</h3>
          <p>查看详细分析报告，获取个性化健康养生建议</p>
        </div>
      </div>
    </div>
  </section>

  <!-- ===== CTA 区域 ===== -->
  <section class="tc-cta-section" style="background-image: url('{CTA_BG_IMG}');">
    <div class="tc-cta-overlay"></div>
    <div class="tc-cta-inner">
      <div class="tc-cta-content">
        <h2 class="tc-cta-title tc-text-gradient">准备好了解您的健康密码了吗？</h2>
        <p class="tc-cta-desc">只需一张舌象照片，30 秒开启您的中医智能健康分析之旅</p>
        <button class="tc-btn-primary tc-btn-large" onclick="showDiagnosisPage()">
          <span class="tc-btn-text">立即开始评测</span>
          <span class="tc-btn-icon">→</span>
        </button>
        <p class="tc-cta-note">⚠️ 本系统为教育演示用途，非医疗器械，不用于临床诊断</p>
      </div>
      <div class="tc-cta-decoration">
        <div class="tc-cta-circle tc-cta-c1"></div>
        <div class="tc-cta-circle tc-cta-c2"></div>
        <div class="tc-cta-circle tc-cta-c3"></div>
      </div>
    </div>
  </section>

</div>
""".format(
        HERO_IMG=hero_img,
        FEATURE_SEG_IMG=feature_seg_img,
        FEATURE_ANALYSIS_IMG=feature_analysis_img,
        FEATURE_CONSTITUTION_IMG=feature_constitution_img,
        FEATURE_ADVICE_IMG=feature_advice_img,
        FEATURE_KNOWLEDGE_IMG=feature_knowledge_img,
        FEATURE_PRIVACY_IMG=feature_privacy_img,
        TECH_AI_IMG=tech_ai_img,
        CTA_BG_IMG=cta_bg_img,
    )


def get_landing_js() -> str:
    """
    返回双页面切换所需的 JavaScript 代码（纯 JS，不含 <script> 标签）。

    注意：在 Gradio 中，gr.HTML() 通过 innerHTML 注入内容，
    浏览器不会执行 innerHTML 中的 <script> 标签。
    因此必须通过 gr.Blocks(head=...) 注入 JS 才能真正运行。

    实现说明：直接用 JS 操作内联 style（带 !important），
    避免 Gradio 内部样式覆盖导致页面空白。
    """
    return r"""
function tcFindPage(id) {
    // Gradio 4.x: elem_id 通常在 form 或 div 元素上
    var el = document.getElementById(id);
    if (el) return el;
    var gradioApp = document.querySelector('gradio-app');
    if (gradioApp && gradioApp.shadowRoot) {
        el = gradioApp.shadowRoot.querySelector('#' + id);
        if (el) return el;
    }
    return null;
}
function tcEnsureVisible(el) {
    if (!el) return;
    // 显示目标元素本身
    el.style.setProperty('display', 'block', 'important');
    // 确保所有祖先元素可见
    var p = el.parentElement;
    while (p && p.tagName !== 'BODY' && p.tagName !== 'HTML') {
        if (getComputedStyle(p).display === 'none') {
            p.style.setProperty('display', 'block', 'important');
        }
        p = p.parentElement;
    }
}
function tcHide(el) {
    if (!el) return;
    el.style.setProperty('display', 'none', 'important');
}
function showDiagnosisPage() {
    var homePage = tcFindPage('tc-page-home');
    var diagPage = tcFindPage('tc-page-diag');
    console.log('[TongueAI] showDiagnosisPage called', {homePage: homePage, diagPage: diagPage});
    tcHide(homePage);
    tcEnsureVisible(diagPage);
    window.scrollTo(0, 0);
}
function showHomePage() {
    var homePage = tcFindPage('tc-page-home');
    var diagPage = tcFindPage('tc-page-diag');
    console.log('[TongueAI] showHomePage called', {homePage: homePage, diagPage: diagPage});
    tcHide(diagPage);
    tcEnsureVisible(homePage);
    window.scrollTo(0, 0);
}
window.addEventListener('DOMContentLoaded', function() {
    console.log('[TongueAI] DOM ready, initializing page state');
    var diagPage = tcFindPage('tc-page-diag');
    tcHide(diagPage);
    var homePage = tcFindPage('tc-page-home');
    tcEnsureVisible(homePage);
});

// ===== 报告打印函数（通过 iframe 隔离，只打印报告内容）=====
function printTongueReport() {
    var reportEl = document.querySelector('.report');
    if (!reportEl) {
        alert('请先生成分析报告');
        return;
    }

    // 收集页面中所有 <style> 标签的内容（包含报告样式）
    var styleCSS = '';
    document.querySelectorAll('style').forEach(function(s) {
        styleCSS += s.innerHTML + '\n';
    });

    // 创建隐藏 iframe
    var iframe = document.createElement('iframe');
    iframe.style.cssText = 'position:fixed;left:-9999px;top:0;width:800px;height:1130px;border:none;';
    document.body.appendChild(iframe);

    var doc = iframe.contentWindow.document;
    doc.open();
    doc.write('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">');
    doc.write('<meta name="viewport" content="width=device-width, initial-scale=1.0">');
    doc.write('<title>TongueAI Pro · 舌象分析报告</title>');
    doc.write('<style>' + styleCSS + '</style>');
    doc.write('</head><body>');
    doc.write(reportEl.outerHTML);
    doc.write('</body></html>');
    doc.close();

    // 等待图片加载后打印
    setTimeout(function() {
        try {
            iframe.contentWindow.focus();
            iframe.contentWindow.print();
        } catch(e) {
            console.error('[TongueAI] Print error:', e);
        }
        // 打印后移除 iframe
        setTimeout(function() {
            if (iframe.parentNode) {
                document.body.removeChild(iframe);
            }
        }, 3000);
    }, 1000);
}

// ===== 报告下载函数（生成独立 HTML 文件并触发下载）=====
function downloadReportHTML() {
    var reportEl = document.querySelector('.report');
    if (!reportEl) {
        alert('请先生成分析报告');
        return;
    }

    // 收集页面中所有 <style> 标签的内容（包含报告样式）
    var styleCSS = '';
    document.querySelectorAll('style').forEach(function(s) {
        styleCSS += s.innerHTML + '\n';
    });

    // 构建独立 HTML 文档（隐藏所有 .no-print 元素，确保下载的是纯报告）
    var html = '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">';
    html += '<meta name="viewport" content="width=device-width, initial-scale=1.0">';
    html += '<title>TongueAI Pro · 舌象分析报告</title>';
    html += '<style>' + styleCSS + '\n.no-print{display:none !important;}</style>';
    html += '</head><body>';
    html += reportEl.outerHTML;
    html += '</body></html>';

    // 通过 Blob 触发浏览器下载
    var blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    var ts = new Date();
    var pad = function(n) { return n < 10 ? '0' + n : '' + n; };
    var fname = 'TongueAI_Report_' + ts.getFullYear() + pad(ts.getMonth()+1) + pad(ts.getDate())
              + '_' + pad(ts.getHours()) + pad(ts.getMinutes()) + pad(ts.getSeconds()) + '.html';
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    setTimeout(function() {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 1000);
}
"""


def get_diagnosis_header_html() -> str:
    """
    生成诊断页顶部导航栏 HTML（含品牌标识和返回首页按钮）
    """
    return r"""
<!-- ===== 诊断页顶部导航栏 ===== -->
<div class="tc-diag-header">
  <div class="tc-diag-header-inner">
    <div class="tc-diag-brand" onclick="window.location.href='/'" title="返回首页">
      <div class="tc-diag-brand-icon">
        <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="20" cy="20" r="18" stroke="url(#diagBrandGrad)" stroke-width="2"/>
          <path d="M20 8C14 8 10 14 10 20C10 28 14 32 20 32C26 32 30 28 30 20C30 14 26 8 20 8Z" fill="url(#diagBrandGrad)" opacity="0.3"/>
          <path d="M20 12C16 12 13 15.5 13 20C13 24.5 16 28 20 28C24 28 27 24.5 27 20C27 15.5 24 12 20 12Z" fill="url(#diagBrandGrad)"/>
          <defs>
            <linearGradient id="diagBrandGrad" x1="0" y1="0" x2="40" y2="40">
              <stop offset="0%" stop-color="#00F5FF"/>
              <stop offset="100%" stop-color="#FF6B35"/>
            </linearGradient>
          </defs>
        </svg>
      </div>
      <span class="tc-diag-brand-name">TongueAI<span class="tc-diag-brand-dot">·</span>Pro</span>
    </div>
    <div class="tc-diag-divider"></div>
    <div class="tc-diag-page-title">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:6px;">
        <circle cx="12" cy="12" r="10"/>
        <circle cx="12" cy="12" r="6"/>
        <circle cx="12" cy="12" r="2"/>
      </svg>
      智能舌象分析
    </div>
    <button class="tc-diag-back-btn" onclick="window.location.href='/'">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M19 12H5M12 19l-7-7 7-7"/>
      </svg>
      返回首页
    </button>
  </div>
</div>
"""


def get_landing_css() -> str:
    """
    生成商业化科技风格主页的 CSS 样式
    """
    return r"""
/* ===== 商业化科技风主页样式 ===== */

.tc-landing {
  position: relative;
  width: 100%;
  min-height: 100vh;
  background: #0a0e1a;
  color: #e8ecf4 !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  overflow: hidden;
}

/* ===== 背景层 ===== */
.tc-bg-grid {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image:
    linear-gradient(rgba(0, 245, 255, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 245, 255, 0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  pointer-events: none;
  z-index: 0;
}

.tc-bg-glow {
  position: absolute;
  border-radius: 50%;
  filter: blur(120px);
  opacity: 0.4;
  pointer-events: none;
  z-index: 0;
}

.tc-glow-1 {
  width: 500px;
  height: 500px;
  top: -100px;
  left: -100px;
  background: radial-gradient(circle, rgba(0, 245, 255, 0.3) 0%, transparent 70%);
  animation: floatGlow 20s ease-in-out infinite;
}

.tc-glow-2 {
  width: 600px;
  height: 600px;
  top: 20%;
  right: -150px;
  background: radial-gradient(circle, rgba(255, 107, 53, 0.25) 0%, transparent 70%);
  animation: floatGlow 25s ease-in-out infinite reverse;
}

.tc-glow-3 {
  width: 400px;
  height: 400px;
  bottom: 10%;
  left: 30%;
  background: radial-gradient(circle, rgba(124, 77, 255, 0.2) 0%, transparent 70%);
  animation: floatGlow 30s ease-in-out infinite;
}

@keyframes floatGlow {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(30px, -30px) scale(1.05); }
  66% { transform: translate(-20px, 20px) scale(0.95); }
}

.tc-noise {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0.03;
  pointer-events: none;
  z-index: 1;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
}

/* ===== 导航栏 ===== */
.tc-nav {
  position: relative;
  z-index: 100;
  padding: 20px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  backdrop-filter: blur(20px);
  background: rgba(10, 14, 26, 0.6);
}

.tc-nav-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.tc-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.tc-brand-icon {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.tc-brand-icon svg {
  width: 40px;
  height: 40px;
}

.tc-brand-name {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 0.5px;
  background: linear-gradient(135deg, #00F5FF, #FF6B35);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.tc-brand-dot {
  color: #00F5FF;
  -webkit-text-fill-color: #00F5FF;
}

.tc-nav-links {
  display: flex;
  align-items: center;
  gap: 32px;
}

.tc-nav-link {
  color: rgba(255, 255, 255, 0.85) !important;
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: color 0.3s ease;
  cursor: pointer;
}

.tc-nav-link:hover {
  color: #00F5FF;
}

.tc-nav-cta {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(0, 245, 255, 0.05));
  border: 1px solid rgba(0, 245, 255, 0.3);
  border-radius: 8px;
  color: #00F5FF;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tc-nav-cta:hover {
  background: rgba(0, 245, 255, 0.15);
  border-color: rgba(0, 245, 255, 0.5);
  box-shadow: 0 0 20px rgba(0, 245, 255, 0.2);
}

/* ===== 英雄区 ===== */
.tc-hero {
  position: relative;
  z-index: 10;
  max-width: 1200px;
  margin: 0 auto;
  padding: 80px 40px 60px;
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 60px;
  align-items: center;
}

.tc-hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: rgba(0, 245, 255, 0.08);
  border: 1px solid rgba(0, 245, 255, 0.2);
  border-radius: 100px;
  font-size: 13px;
  color: #00F5FF;
  margin-bottom: 24px;
}

.tc-hero-badge-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #00F5FF;
  box-shadow: 0 0 8px #00F5FF;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.tc-hero-title {
  font-size: 56px;
  font-weight: 800;
  line-height: 1.15;
  margin: 0 0 20px;
  letter-spacing: -1px;
}

.tc-hero-title-line {
  display: block;
}

.tc-hero-title-gradient {
  background: linear-gradient(135deg, #00F5FF 0%, #B388FF 50%, #FF6B35 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
}

.tc-hero-desc {
  font-size: 18px;
  line-height: 1.7;
  color: rgba(255, 255, 255, 0.8) !important;
  margin: 0 0 36px;
}

.tc-hero-actions {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 48px;
}

.tc-btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 16px 32px;
  background: linear-gradient(135deg, #00F5FF, #0091EA);
  border: none;
  border-radius: 12px;
  color: #0a0e1a;
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(0, 245, 255, 0.3);
}

.tc-btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(0, 245, 255, 0.4);
}

.tc-btn-large {
  padding: 20px 48px;
  font-size: 18px;
}

.tc-btn-icon {
  transition: transform 0.3s ease;
}

.tc-btn-primary:hover .tc-btn-icon {
  transform: translateX(4px);
}

.tc-btn-ghost {
  display: inline-flex;
  align-items: center;
  padding: 16px 28px;
  background: transparent;
  border: 1px solid rgba(232, 236, 244, 0.2);
  border-radius: 12px;
  color: #e8ecf4;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  text-decoration: none;
  transition: all 0.3s ease;
}

.tc-btn-ghost:hover {
  border-color: rgba(232, 236, 244, 0.4);
  background: rgba(232, 236, 244, 0.05);
}

.tc-hero-stats {
  display: flex;
  align-items: center;
  gap: 32px;
  padding-top: 32px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.tc-stat {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.tc-stat-num {
  font-size: 32px;
  font-weight: 800;
  background: linear-gradient(135deg, #00F5FF, #FF6B35);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.tc-stat-num span {
  font-size: 18px;
  font-weight: 600;
}

.tc-stat-label {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.8) !important;
}

.tc-stat-divider {
  width: 1px;
  height: 40px;
  background: rgba(255, 255, 255, 0.08);
}

/* ===== 英雄区视觉 ===== */
.tc-hero-visual {
  position: relative;
  height: 480px;
}

.tc-hero-card {
  background: rgba(20, 25, 40, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  backdrop-filter: blur(20px);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.tc-hero-card-main {
  position: absolute;
  top: 0;
  left: 10%;
  width: 280px;
  z-index: 2;
  animation: floatCard 6s ease-in-out infinite;
}

@keyframes floatCard {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-15px); }
}

.tc-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 18px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.tc-card-dots {
  display: flex;
  gap: 6px;
}

.tc-card-dots span {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
}

.tc-card-dots span:first-child {
  background: #FF6B35;
}

.tc-card-dots span:nth-child(2) {
  background: #FFD54F;
}

.tc-card-dots span:nth-child(3) {
  background: #69F0AE;
}

.tc-card-title {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.75) !important;
  margin-left: auto;
}

.tc-card-body {
  padding: 24px;
  display: flex;
  justify-content: center;
}

.tc-tongue-visual {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 280px;
  overflow: hidden;
  border-radius: 8px;
}

.tc-tongue-visual svg {
  width: 100%;
  height: 100%;
}

.tc-hero-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.tc-detect-point {
  position: absolute;
  width: 10px;
  height: 10px;
  transform: translate(-50%, -50%);
}

.tc-detect-pulse {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: #00F5FF;
  animation: detectPulse 2s ease-out infinite;
}

@keyframes detectPulse {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  100% {
    transform: scale(3);
    opacity: 0;
  }
}

.tc-hero-card-float {
  position: absolute;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 18px;
  z-index: 3;
}

.tc-float-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.tc-float-value {
  font-size: 18px;
  font-weight: 700;
  color: #fff;
}

.tc-float-label {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.7) !important;
}

.tc-float-1 {
  top: 40px;
  right: 0;
  animation: floatCard 5s ease-in-out infinite;
  animation-delay: -1s;
}

.tc-float-2 {
  top: 180px;
  right: -20px;
  animation: floatCard 7s ease-in-out infinite;
  animation-delay: -2s;
}

.tc-float-3 {
  bottom: 60px;
  right: 20px;
  animation: floatCard 6s ease-in-out infinite;
  animation-delay: -3s;
}

/* ===== 信任背书 ===== */
.tc-trust {
  position: relative;
  z-index: 10;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  background: rgba(255, 255, 255, 0.01);
}

.tc-trust-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 32px 40px;
  display: flex;
  justify-content: space-around;
  align-items: center;
  flex-wrap: wrap;
  gap: 24px;
}

.tc-trust-item {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 14px;
  font-weight: 500;
}

.tc-trust-item span {
  background: linear-gradient(135deg, #00F5FF 0%, #69F0AE 50%, #FFD54F 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
  font-weight: 600;
  letter-spacing: 0.3px;
}

/* ===== 通用区块 ===== */
.tc-section {
  position: relative;
  z-index: 10;
  padding: 100px 40px;
}

.tc-section-inner {
  max-width: 1200px;
  margin: 0 auto;
}

.tc-section-alt {
  background: rgba(255, 255, 255, 0.015);
}

.tc-section-header {
  text-align: center;
  margin-bottom: 60px;
}

.tc-section-tag {
  display: inline-block;
  padding: 6px 16px;
  background: rgba(0, 245, 255, 0.08);
  border: 1px solid rgba(0, 245, 255, 0.2);
  border-radius: 100px;
  font-size: 13px;
  color: #00F5FF;
  margin-bottom: 16px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.tc-section-title {
  font-size: 42px;
  font-weight: 800;
  margin: 0 0 16px;
  letter-spacing: -0.5px;
}

.tc-text-gradient {
  background: linear-gradient(135deg, #00F5FF 0%, #FF6B35 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
}

.tc-section-desc {
  font-size: 16px;
  color: rgba(255, 255, 255, 0.75) !important;
  margin: 0;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

/* ===== 功能卡片 ===== */
.tc-features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}

.tc-feature-card {
  padding: 0;
  background: rgba(20, 25, 40, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 16px;
  transition: all 0.4s ease;
  backdrop-filter: blur(10px);
  overflow: hidden;
}

.tc-feature-card:hover {
  transform: translateY(-6px);
  border-color: rgba(0, 245, 255, 0.2);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.tc-feature-img-wrap {
  position: relative;
  width: 100%;
  height: 180px;
  overflow: hidden;
}

.tc-feature-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.5s ease;
}

.tc-feature-card:hover .tc-feature-img {
  transform: scale(1.08);
}

.tc-feature-img-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(15, 18, 30, 0.1) 0%, rgba(15, 18, 30, 0.8) 100%);
}

.tc-feature-card .tc-feature-icon,
.tc-feature-card .tc-feature-title,
.tc-feature-card .tc-feature-desc,
.tc-feature-card .tc-feature-tags {
  margin-left: 28px;
  margin-right: 28px;
}

.tc-feature-card .tc-feature-icon {
  margin-top: -28px;
  position: relative;
  z-index: 2;
}

.tc-feature-icon {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20px;
  border: 1px solid;
}

.tc-feature-title {
  font-size: 18px;
  font-weight: 700;
  margin: 0 0 12px;
}

.tc-feature-desc {
  font-size: 14px;
  line-height: 1.7;
  color: rgba(255, 255, 255, 0.7) !important;
  margin: 0 0 20px;
}

.tc-feature-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 28px !important;
}

.tc-feature-tags span {
  padding: 4px 10px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.75) !important;
}

/* ===== 技术优势 ===== */
.tc-tech-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  align-items: center;
}

.tc-tech-visual {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  aspect-ratio: 4 / 3;
}

.tc-tech-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.tc-tech-visual-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 24px;
  display: flex;
  gap: 32px;
  background: linear-gradient(0deg, rgba(15, 18, 30, 0.9) 0%, transparent 100%);
}

.tc-tech-stat-num {
  font-size: 28px;
  font-weight: 800;
  background: linear-gradient(135deg, #00F5FF, #FF6B35);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.tc-tech-stat-label {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.75) !important;
  margin-top: 2px;
}

.tc-tech-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.tc-tech-item {
  padding: 28px;
  background: rgba(20, 25, 40, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  transition: all 0.3s ease;
}

.tc-tech-item:hover {
  border-color: rgba(255, 107, 53, 0.3);
  background: rgba(20, 25, 40, 0.5);
}

.tc-tech-num {
  font-size: 36px;
  font-weight: 800;
  background: linear-gradient(135deg, rgba(0, 245, 255, 0.5), rgba(0, 245, 255, 0.1));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 12px;
}

.tc-tech-item h3 {
  font-size: 16px;
  font-weight: 700;
  margin: 0 0 8px;
}

.tc-tech-item p {
  font-size: 13px;
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.7) !important;
  margin: 0;
}

/* ===== 使用流程 ===== */
.tc-steps {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  flex-wrap: wrap;
}

.tc-step {
  flex: 1;
  min-width: 200px;
  max-width: 280px;
  padding: 36px 28px;
  text-align: center;
  background: rgba(20, 25, 40, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 16px;
  position: relative;
}

.tc-step-num {
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  padding: 4px 14px;
  background: linear-gradient(135deg, #00F5FF, #0091EA);
  border-radius: 100px;
  font-size: 13px;
  font-weight: 700;
  color: #0a0e1a;
}

.tc-step-icon {
  width: 72px;
  height: 72px;
  margin: 12px auto 20px;
  background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(0, 145, 234, 0.05));
  border: 1px solid rgba(0, 245, 255, 0.2);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #00F5FF;
}

.tc-step:nth-child(3) .tc-step-icon {
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(255, 61, 0, 0.05));
  border-color: rgba(255, 107, 53, 0.2);
  color: #FF6B35;
}

.tc-step:nth-child(5) .tc-step-icon {
  background: linear-gradient(135deg, rgba(179, 136, 255, 0.1), rgba(124, 77, 255, 0.05));
  border-color: rgba(179, 136, 255, 0.2);
  color: #B388FF;
}

.tc-step h3 {
  font-size: 18px;
  font-weight: 700;
  margin: 0 0 8px;
  background: linear-gradient(135deg, #00F5FF 0%, #69F0AE 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
}

.tc-step:nth-child(3) h3 {
  background: linear-gradient(135deg, #FF6B35 0%, #FFD54F 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
}

.tc-step:nth-child(5) h3 {
  background: linear-gradient(135deg, #B388FF 0%, #F06292 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
}

.tc-step p {
  font-size: 14px;
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.7) !important;
  margin: 0;
}

.tc-step-arrow {
  flex-shrink: 0;
  color: #00F5FF;
}

/* ===== CTA 区域 ===== */
.tc-cta-section {
  position: relative;
  z-index: 10;
  padding: 120px 40px;
  overflow: hidden;
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}

.tc-cta-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(15, 18, 30, 0.92) 0%, rgba(15, 18, 30, 0.75) 50%, rgba(15, 18, 30, 0.92) 100%);
  z-index: 1;
}

.tc-cta-inner {
  max-width: 900px;
  margin: 0 auto;
  text-align: center;
  position: relative;
}

.tc-cta-content {
  position: relative;
  z-index: 2;
}

.tc-cta-title {
  font-size: 40px;
  font-weight: 800;
  margin: 0 0 16px;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #00F5FF 0%, #B388FF 40%, #FF6B35 80%, #FFD54F 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
  filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.3));
}

.tc-cta-desc {
  font-size: 17px;
  color: rgba(255, 255, 255, 0.8) !important;
  margin: 0 0 32px;
}

.tc-cta-note {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.6) !important;
  margin: 20px 0 0;
}

.tc-cta-decoration {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 500px;
  height: 500px;
  pointer-events: none;
}

.tc-cta-circle {
  position: absolute;
  border-radius: 50%;
  border: 1px solid rgba(0, 245, 255, 0.1);
}

.tc-cta-c1 {
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  animation: rotateCircle 30s linear infinite;
}

.tc-cta-c2 {
  width: 70%;
  height: 70%;
  top: 15%;
  left: 15%;
  animation: rotateCircle 20s linear infinite reverse;
  border-color: rgba(255, 107, 53, 0.1);
}

.tc-cta-c3 {
  width: 40%;
  height: 40%;
  top: 30%;
  left: 30%;
  animation: rotateCircle 15s linear infinite;
  border-color: rgba(179, 136, 255, 0.15);
}

@keyframes rotateCircle {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* ===== 响应式 ===== */
@media (max-width: 900px) {
  .tc-hero {
    grid-template-columns: 1fr;
    padding: 60px 24px 40px;
    gap: 40px;
  }
  
  .tc-hero-title {
    font-size: 40px;
  }
  
  .tc-hero-visual {
    height: 380px;
    order: -1;
  }
  
  .tc-hero-card-main {
    left: 50%;
    transform: translateX(-50%);
  }
  
  .tc-features-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .tc-tech-layout {
    grid-template-columns: 1fr;
  }
  
  .tc-tech-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .tc-steps {
    flex-direction: column;
  }
  
  .tc-step-arrow {
    transform: rotate(90deg);
  }
  
  .tc-section {
    padding: 60px 24px;
  }
  
  .tc-section-title {
    font-size: 32px;
  }
  
  .tc-nav-links {
    display: none;
  }
  
  .tc-nav-inner {
    padding: 0 24px;
  }
}

@media (max-width: 600px) {
  .tc-features-grid {
    grid-template-columns: 1fr;
  }
  
  .tc-tech-grid {
    grid-template-columns: 1fr;
  }
  
  .tc-hero-stats {
    flex-wrap: wrap;
    gap: 20px;
  }
  
  .tc-stat-divider {
    display: none;
  }
  
  .tc-cta-title {
    font-size: 28px;
  }
}

/* ======================================== */
/* ===== 双页面切换布局 ===== */
/* ======================================== */

/* 页面容器：覆盖 Gradio 默认样式 */
#tc-page-home, #tc-page-diag {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    gap: 0 !important;
    background: transparent !important;
}

/* 默认隐藏诊断页（由 JS 接管控制，避免与内联 style 冲突） */
#tc-page-diag {
    display: none;
}

/* 切换到诊断页时：隐藏首页、显示诊断页（由 JS 直接操作内联 style，此处保留作为兜底） */
body.tc-view-diag #tc-page-home {
    display: none;
}
body.tc-view-diag #tc-page-diag {
    display: block;
}

/* 切换到诊断页时：隐藏 Gradio 默认的 footer */
body.tc-view-diag .gradio-container > .footer {
    display: none !important;
}

/* ===== 诊断页顶部导航栏 ===== */
.tc-diag-header {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: rgba(10, 14, 26, 0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(0, 245, 255, 0.15);
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.2);
}

.tc-diag-header-inner {
    max-width: 1200px;
    margin: 0 auto;
    padding: 12px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
}

.tc-diag-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    transition: opacity 0.3s ease;
}

.tc-diag-brand:hover {
    opacity: 0.8;
}

.tc-diag-brand-icon {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.tc-diag-brand-icon svg {
    width: 32px;
    height: 32px;
}

.tc-diag-brand-name {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
    background: linear-gradient(135deg, #00F5FF, #FF6B35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.tc-diag-brand-dot {
    color: #00F5FF;
    -webkit-text-fill-color: #00F5FF;
}

.tc-diag-divider {
    width: 1px;
    height: 24px;
    background: rgba(255, 255, 255, 0.15);
}

.tc-diag-page-title {
    color: rgba(232, 236, 244, 0.85);
    font-size: 16px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

.tc-diag-back-btn {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 20px;
    background: linear-gradient(135deg, rgba(0, 245, 255, 0.12), rgba(0, 245, 255, 0.05));
    border: 1px solid rgba(0, 245, 255, 0.3);
    border-radius: 10px;
    color: #00F5FF;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

.tc-diag-back-btn:hover {
    background: rgba(0, 245, 255, 0.18);
    border-color: rgba(0, 245, 255, 0.5);
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.25);
    transform: translateY(-1px);
}

.tc-diag-back-btn:active {
    transform: translateY(0);
}

/* ===== 诊断页内容区域 ===== */
.tc-diag-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 40px;
}

/* 诊断页渐变标题栏 */
.tc-diag-hero-bar {
    background: linear-gradient(135deg, rgba(0, 245, 255, 0.08), rgba(255, 107, 53, 0.05));
    border: 1px solid rgba(0, 245, 255, 0.15);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    text-align: center;
}

.tc-diag-hero-bar h2 {
    font-size: 28px;
    font-weight: 800;
    margin: 0 0 8px;
    background: linear-gradient(135deg, #00F5FF, #FF6B35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.tc-diag-hero-bar p {
    color: rgba(0, 0, 0, 0.55);
    margin: 0;
    font-size: 15px;
}

/* 响应式 */
@media (max-width: 768px) {
    .tc-diag-header-inner {
        padding: 10px 20px;
    }
    .tc-diag-page-title {
        display: none;
    }
    .tc-diag-divider {
        display: none;
    }
    .tc-diag-content {
        padding: 20px;
    }
    .tc-diag-hero-bar {
        padding: 20px;
    }
    .tc-diag-hero-bar h2 {
        font-size: 22px;
    }
}
"""
