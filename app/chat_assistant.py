"""
智能解读对话助手模块
====================
提供基于舌象检测和综合分析报告上下文的 AI 对话功能。

功能：
1. 会话管理：保存综合报告数据，供对话使用
2. 系统提示构建：将报告数据转化为 AI 医生知识库
3. Agnes AI 对话：支持多轮对话，上下文连续
4. 对话页面 HTML：温馨医学风格 UI

路由：
- GET  /chat           → 对话页面
- POST /chat/api/message → 对话 API
"""

import json
import uuid
import urllib.request
import urllib.error
import os
import base64

from .llm import (
    resolve_api_key,
    AGNES_CHAT_ENDPOINT,
    AGNES_DEFAULT_MODEL,
    REQUEST_TIMEOUT,
)


# ============================================================
# 会话管理
# ============================================================

_chat_sessions = {}


def create_session(report_data: dict, image_b64: str = None) -> str:
    """
    创建对话会话，保存报告数据。

    参数:
        report_data: 综合分析结果（_last_result）
        image_b64: 舌象图片的 base64 编码（可选）

    返回:
        session_id: 会话 ID（用于 URL 查询参数）
    """
    session_id = uuid.uuid4().hex[:16]
    _chat_sessions[session_id] = {
        "report": report_data,
        "image_b64": image_b64,
        "history": [],
    }
    return session_id


def get_session(session_id: str) -> dict:
    """获取会话数据，不存在返回空字典"""
    return _chat_sessions.get(session_id, {})


# ============================================================
# 系统提示构建
# ============================================================

def build_system_prompt(report_data: dict) -> str:
    """
    根据综合分析报告构建 AI 医生的系统提示词。

    将舌质、舌苔、21类检测结果、AI评语、问卷数据等
    转化为结构化的知识库上下文。
    """
    body = report_data.get("tongue_body", {})
    coating = report_data.get("coating", {})
    features = report_data.get("features", {})
    yolo_labels = report_data.get("yolo_labels", [])
    ai_commentary = report_data.get("ai_commentary", "")
    questionnaire = report_data.get("questionnaire_answers", {})

    # 舌质信息
    body_name = body.get("name", "未知")
    body_tcm = body.get("tcm_meaning", "")
    body_desc = body.get("description", "")
    body_color = body.get("color_info", {})

    # 舌苔信息
    coating_name = coating.get("name", "未知")
    coating_tcm = coating.get("tcm_meaning", "")
    coating_desc = coating.get("description", "")

    # 21类检测结果
    yolo_names = [lbl.get("name", "") for lbl in yolo_labels]

    # 体质
    constitution = report_data.get("constitution", {})
    const_type = constitution.get("type", "未知")
    const_feature = constitution.get("feature", "")

    # 问卷数据
    questionnaire_text = ""
    if questionnaire:
        try:
            from .questionnaire import format_questionnaire_for_llm
            questionnaire_text = format_questionnaire_for_llm(questionnaire)
        except Exception:
            questionnaire_text = str(questionnaire)

    prompt = f"""你是一位经验丰富的中医AI医生助手，名叫"舌诊小助手"。你的任务是根据用户的舌象检测报告，用通俗易懂的语言解答用户的健康疑问。

## 用户舌象检测报告

### 舌质分析
- 舌质类型：{body_name}
- 中医含义：{body_tcm}
- 描述：{body_desc}

### 舌苔分析
- 舌苔类型：{coating_name}
- 中医含义：{coating_tcm}
- 描述：{coating_desc}

### 21类检测结果
{", ".join(yolo_names) if yolo_names else "无"}

### 体质辨识
- 体质类型：{const_type}
- 体质特征：{const_feature}

### AI综合评语
{ai_commentary if ai_commentary else "未生成"}

### 问卷数据
{questionnaire_text if questionnaire_text else "未填写"}

## 对话规则
1. 基于上述检测报告数据回答用户问题，不要编造报告中没有的数据
2. 用通俗易懂的语言解释中医术语，让普通人也能理解
3. 如果用户问的问题超出了舌诊范围，诚实地告知并建议咨询专业医生
4. 每次回复控制在300字以内，重点突出，避免长篇大论
5. 语气温馨亲切，像一位关心你的家庭医生
6. 对于健康焦虑的用户，给予安抚和正向引导
7. 始终提醒：AI分析仅供参考，不构成医疗诊断，如有不适请就医
8. 可以适当使用表情符号增加亲和力，但不要过度
"""
    return prompt


# ============================================================
# Agnes AI 对话调用
# ============================================================

def chat_with_agnes(
    message: str,
    history: list,
    session_id: str,
    api_key: str = None,
    model: str = None,
) -> str:
    """
    调用 Agnes AI 进行多轮对话。

    参数:
        message: 用户消息
        history: 对话历史 [{"user": "...", "bot": "..."}]
        session_id: 会话 ID
        api_key: API Key（为空则从文件加载）
        model: 模型名称（为空则使用默认）

    返回:
        AI 回复文本
    """
    session = get_session(session_id)
    if not session:
        return "未找到报告数据，请返回主页先生成综合报告，再进入对话咨询。"

    # 解析 API Key
    if not api_key:
        try:
            api_key = resolve_api_key()
        except RuntimeError:
            return "未配置 Agnes AI API Key，无法进行对话。请在主页面设置 API Key。"

    if not model:
        model = "agnes-2.5-flash"

    # 构建系统提示
    system_prompt = build_system_prompt(session["report"])

    # 构建消息列表
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]:
        messages.append({"role": "user", "content": h.get("user", "")})
        messages.append({"role": "assistant", "content": h.get("bot", "")})
    messages.append({"role": "user", "content": message})

    # 调用 API
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 2000,
        "top_p": 0.9,
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
        return f"AI 对话出错（HTTP {e.code}），请稍后重试。"
    except Exception as e:
        err_lower = str(e).lower()
        if "timed out" in err_lower or "timeout" in type(e).__name__.lower():
            # 超时后降级到 flash 模型重试
            if model != "agnes-2.5-flash":
                return chat_with_agnes(message, history, session_id, api_key, "agnes-2.5-flash")
            return "AI 回复超时，请稍后重试。"
        return f"AI 对话出错：{str(e)[:100]}"

    # 解析响应
    try:
        choices = result.get("choices", [])
        if not choices:
            return "AI 未返回有效回复，请重试。"
        content = choices[0].get("message", {}).get("content", "")
        if content.strip():
            return content.strip()
        return "AI 正在思考中但未生成回复，请重新提问。"
    except Exception:
        return "AI 回复解析失败，请重试。"


# ============================================================
# 对话页面 HTML
# ============================================================

def _get_avatar_base64() -> str:
    """读取 AI 医生头像并转为 base64"""
    avatar_path = os.path.join(
        os.path.dirname(__file__), "static", "images", "ai_doctor_avatar.jpg"
    )
    if os.path.exists(avatar_path):
        with open(avatar_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def get_chat_page_html(session_id: str = "") -> str:
    """返回对话页面完整 HTML"""
    avatar_b64 = _get_avatar_base64()
    avatar_src = f"data:image/jpeg;base64,{avatar_b64}" if avatar_b64 else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>舌诊小助手 - AI 智能解读咨询</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

:root {{
  --primary: #2A9D8F;
  --primary-light: #E0F5F2;
  --primary-dark: #1E7268;
  --accent: #F4A261;
  --accent-light: #FEF3E7;
  --bg: #F5F9F8;
  --card-bg: #FFFFFF;
  --text: #2D3436;
  --text-light: #636E72;
  --text-muted: #B2BEC3;
  --border: #E3E8E6;
  --shadow: 0 2px 12px rgba(42, 157, 143, 0.08);
  --shadow-hover: 0 4px 20px rgba(42, 157, 143, 0.15);
  --danger: #E76F51;
  --success: #27AE60;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

/* ===== 顶部导航栏 ===== */
.navbar {{
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  z-index: 10;
  flex-shrink: 0;
}}

.navbar-left {{
  display: flex;
  align-items: center;
  gap: 12px;
}}

.navbar-avatar {{
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: 2px solid rgba(255,255,255,0.3);
  object-fit: cover;
  background: rgba(255,255,255,0.1);
}}

.navbar-title {{
  color: #fff;
  font-size: 18px;
  font-weight: 600;
}}

.navbar-subtitle {{
  color: rgba(255,255,255,0.8);
  font-size: 12px;
  margin-top: 2px;
}}

.btn-back {{
  background: rgba(255,255,255,0.15);
  color: #fff;
  border: 1px solid rgba(255,255,255,0.3);
  padding: 8px 20px;
  border-radius: 20px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 6px;
  text-decoration: none;
}}

.btn-back:hover {{
  background: rgba(255,255,255,0.25);
  transform: translateX(-2px);
}}

/* ===== 聊天区域 ===== */
.chat-container {{
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 800px;
  width: 100%;
  margin: 0 auto;
}}

.chat-container::-webkit-scrollbar {{
  width: 6px;
}}
.chat-container::-webkit-scrollbar-track {{
  background: transparent;
}}
.chat-container::-webkit-scrollbar-thumb {{
  background: var(--border);
  border-radius: 3px;
}}

/* ===== 欢迎卡片 ===== */
.welcome-card {{
  background: var(--card-bg);
  border-radius: 16px;
  padding: 28px;
  box-shadow: var(--shadow);
  text-align: center;
  margin-bottom: 8px;
}}

.welcome-avatar {{
  width: 80px;
  height: 80px;
  border-radius: 50%;
  border: 3px solid var(--primary-light);
  object-fit: cover;
  margin-bottom: 16px;
}}

.welcome-title {{
  font-size: 22px;
  font-weight: 700;
  color: var(--primary-dark);
  margin-bottom: 8px;
}}

.welcome-desc {{
  font-size: 14px;
  color: var(--text-light);
  line-height: 1.6;
  margin-bottom: 20px;
}}

.welcome-tags {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
}}

.welcome-tag {{
  background: var(--primary-light);
  color: var(--primary-dark);
  padding: 6px 14px;
  border-radius: 16px;
  font-size: 13px;
  font-weight: 500;
}}

/* ===== 消息气泡 ===== */
.message {{
  display: flex;
  gap: 12px;
  max-width: 85%;
  animation: fadeIn 0.3s ease;
}}

@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

.message.user {{
  align-self: flex-end;
  flex-direction: row-reverse;
}}

.message-avatar {{
  width: 36px;
  height: 36px;
  border-radius: 50%;
  flex-shrink: 0;
  object-fit: cover;
}}

.message.user .message-avatar {{
  background: var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 16px;
  font-weight: 600;
}}

.message-content {{
  background: var(--card-bg);
  padding: 14px 18px;
  border-radius: 16px;
  font-size: 15px;
  line-height: 1.7;
  box-shadow: var(--shadow);
  color: var(--text);
}}

.message.user .message-content {{
  background: var(--primary);
  color: #fff;
  border-radius: 16px 16px 4px 16px;
}}

.message.bot .message-content {{
  background: var(--card-bg);
  border-radius: 16px 16px 16px 4px;
  border: 1px solid var(--border);
}}

.message-content p {{ margin-bottom: 8px; }}
.message-content p:last-child {{ margin-bottom: 0; }}
.message-content ul, .message-content ol {{ margin: 8px 0 8px 20px; }}
.message-content strong {{ color: var(--primary-dark); }}
.message.user .message-content strong {{ color: #fff; }}

/* ===== 加载动画 ===== */
.typing-indicator {{
  display: flex;
  gap: 4px;
  padding: 14px 18px;
}}

.typing-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--text-muted);
  animation: typing 1.4s infinite;
}}

.typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}

@keyframes typing {{
  0%, 60%, 100% {{ transform: translateY(0); opacity: 0.4; }}
  30% {{ transform: translateY(-6px); opacity: 1; }}
}}

/* ===== 快捷问题 ===== */
.quick-questions {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}}

.quick-question {{
  background: var(--accent-light);
  color: var(--accent);
  border: 1px solid rgba(244, 162, 97, 0.3);
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}}

.quick-question:hover {{
  background: var(--accent);
  color: #fff;
  transform: translateY(-1px);
}}

/* ===== 输入区域 ===== */
.input-area {{
  padding: 16px 24px;
  background: var(--card-bg);
  border-top: 1px solid var(--border);
  display: flex;
  gap: 12px;
  align-items: flex-end;
  max-width: 800px;
  width: 100%;
  margin: 0 auto;
  flex-shrink: 0;
}}

.input-area textarea {{
  flex: 1;
  border: 1.5px solid var(--border);
  border-radius: 12px;
  padding: 12px 16px;
  font-size: 15px;
  font-family: inherit;
  resize: none;
  min-height: 48px;
  max-height: 120px;
  transition: border-color 0.2s;
  outline: none;
  line-height: 1.5;
}}

.input-area textarea:focus {{
  border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-light);
}}

.btn-send {{
  background: var(--primary);
  color: #fff;
  border: none;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  font-size: 20px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}}

.btn-send:hover {{
  background: var(--primary-dark);
  transform: scale(1.05);
}}

.btn-send:disabled {{
  background: var(--text-muted);
  cursor: not-allowed;
  transform: none;
}}

/* ===== 免责声明 ===== */
.disclaimer {{
  text-align: center;
  font-size: 12px;
  color: var(--text-muted);
  padding: 8px 24px;
  background: var(--card-bg);
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}}

/* ===== 错误消息 ===== */
.message.error .message-content {{
  background: #FEF2F2;
  border-color: #FECACA;
  color: var(--danger);
}}
</style>
</head>
<body>

<!-- 顶部导航栏 -->
<div class="navbar">
  <div class="navbar-left">
    <img src="{avatar_src}" class="navbar-avatar" alt="AI医生"
         onerror="this.style.display='none'">
    <div>
      <div class="navbar-title">舌诊小助手</div>
      <div class="navbar-subtitle">AI 智能解读咨询 · 基于您的舌象报告</div>
    </div>
  </div>
  <a href="/" class="btn-back">
    <span>&larr;</span> 返回主页
  </a>
</div>

<!-- 聊天区域 -->
<div class="chat-container" id="chatContainer">

  <!-- 欢迎卡片 -->
  <div class="welcome-card">
    <img src="{avatar_src}" class="welcome-avatar" alt="AI医生"
         onerror="this.style.display='none'">
    <div class="welcome-title">👋 您好，我是舌诊小助手</div>
    <div class="welcome-desc">
      我已阅读您的舌象检测报告和综合分析结果。<br>
      有任何看不懂的地方，或者想了解的健康问题，都可以问我！
    </div>
    <div class="welcome-tags">
      <span class="welcome-tag">📋 报告解读</span>
      <span class="welcome-tag">🌿 中医调理建议</span>
      <span class="welcome-tag">💡 体质分析</span>
      <span class="welcome-tag">⚠️ 疾病风险</span>
    </div>
  </div>

  <!-- 快捷问题 -->
  <div class="quick-questions" id="quickQuestions">
    <div class="quick-question" onclick="sendQuickMessage('我的舌象检测结果说明什么？请帮我解读一下')">我的舌象检测结果说明什么？</div>
    <div class="quick-question" onclick="sendQuickMessage('我的体质类型是什么？日常生活中需要注意什么？')">我的体质需要注意什么？</div>
    <div class="quick-question" onclick="sendQuickMessage('报告中提到的疾病风险严重吗？我该怎么办？')">疾病风险严重吗？</div>
    <div class="quick-question" onclick="sendQuickMessage('有什么食疗或生活习惯建议可以改善我的体质吗？')">食疗和生活建议</div>
  </div>

</div>

<!-- 输入区域 -->
<div class="input-area">
  <textarea id="messageInput"
            placeholder="输入您的问题..."
            rows="1"
            onkeydown="handleKeyDown(event)"
            oninput="autoResize(this)"></textarea>
  <button class="btn-send" id="sendBtn" onclick="sendMessage()">
    ➤
  </button>
</div>

<!-- 免责声明 -->
<div class="disclaimer">
  ⚠️ AI 对话仅供参考，不构成医疗诊断或治疗建议。如有健康问题，请务必咨询专业医生。
</div>

<script>
const SESSION_ID = "{session_id}";
const AVATAR_SRC = "{avatar_src}";
let history = [];
let isWaiting = false;

// 自动调整输入框高度
function autoResize(textarea) {{
  textarea.style.height = '48px';
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}}

// 处理回车键
function handleKeyDown(event) {{
  if (event.key === 'Enter' && !event.shiftKey) {{
    event.preventDefault();
    sendMessage();
  }}
}}

// HTML 转义
function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

// 简易 Markdown 渲染
function renderMarkdown(text) {{
  let html = escapeHtml(text);
  // 加粗
  html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  // 列表
  html = html.replace(/^[\\*\\-]\\s(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.+<\\/li>)/s, '<ul>$1</ul>');
  // 换行
  html = html.replace(/\\n/g, '<br>');
  // 段落
  html = html.replace(/<br><br>/g, '</p><p>');
  html = '<p>' + html + '</p>';
  return html;
}}

// 发送快捷问题
function sendQuickMessage(text) {{
  document.getElementById('messageInput').value = text;
  sendMessage();
}}

// 添加消息到聊天区域
function addMessage(role, content, isError = false) {{
  const container = document.getElementById('chatContainer');
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message ' + role + (isError ? ' error' : '');

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';

  if (role === 'bot') {{
    const img = document.createElement('img');
    img.src = AVATAR_SRC;
    img.className = 'message-avatar';
    img.onerror = function() {{ this.style.display = 'none'; }};
    avatar.replaceWith(img);
  }} else {{
    avatar.textContent = '我';
  }}

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  if (role === 'bot') {{
    contentDiv.innerHTML = renderMarkdown(content);
  }} else {{
    contentDiv.textContent = content;
  }}

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(contentDiv);
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}}

// 显示加载动画
function showTyping() {{
  const container = document.getElementById('chatContainer');
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message bot';
  msgDiv.id = 'typingIndicator';

  const avatar = document.createElement('img');
  avatar.src = AVATAR_SRC;
  avatar.className = 'message-avatar';
  avatar.onerror = function() {{ this.style.display = 'none'; }};

  const typingDiv = document.createElement('div');
  typingDiv.className = 'message-content typing-indicator';
  typingDiv.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(typingDiv);
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}}

// 移除加载动画
function removeTyping() {{
  const indicator = document.getElementById('typingIndicator');
  if (indicator) indicator.remove();
}}

// 发送消息
async function sendMessage() {{
  if (isWaiting) return;

  const input = document.getElementById('messageInput');
  const text = input.value.trim();
  if (!text) return;

  // 添加用户消息
  addMessage('user', text);
  history.push({{ user: text, bot: '' }});

  // 清空输入框
  input.value = '';
  input.style.height = '48px';

  // 禁用发送按钮
  isWaiting = true;
  document.getElementById('sendBtn').disabled = true;

  // 隐藏快捷问题
  const qq = document.getElementById('quickQuestions');
  if (qq) qq.style.display = 'none';

  // 显示加载动画
  showTyping();

  try {{
    const response = await fetch('/chat/api/message', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        message: text,
        history: history.slice(0, -1),
        session_id: SESSION_ID,
      }}),
    }});

    const data = await response.json();
    removeTyping();

    if (data.reply) {{
      addMessage('bot', data.reply);
      history[history.length - 1].bot = data.reply;
    }} else if (data.error) {{
      addMessage('bot', data.error, true);
      history[history.length - 1].bot = data.error;
    }}
  }} catch (err) {{
    removeTyping();
    addMessage('bot', '网络连接出错，请检查网络后重试。', true);
  }}

  isWaiting = false;
  document.getElementById('sendBtn').disabled = false;
  input.focus();
}}

// 页面加载后自动聚焦输入框
window.addEventListener('load', function() {{
  document.getElementById('messageInput').focus();
}});
</script>

</body>
</html>"""
