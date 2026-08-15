#!/usr/bin/env bash
# ============================================================
# 中医AI舌诊辅助健康识别系统 - 一键部署脚本
# 适用环境：腾讯云 Ubuntu 22.04 LTS（轻量应用服务器 / CVM）
# 使用方法：
#   sudo bash deploy.sh                          # 交互式部署（默认配置）
#   sudo bash deploy.sh --api-key=sk-xxx         # 指定 Agnes API Key
#   sudo bash deploy.sh --with-u2net             # 安装 U2-Net 分割后端
#   sudo bash deploy.sh --with-nginx             # 配置 Nginx 反向代理
#   sudo bash deploy.sh --with-u2net --with-nginx --api-key=sk-xxx
# ============================================================
set -euo pipefail

# ---------- 颜色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[信息]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[完成]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[警告]${NC} $*"; }
log_error() { echo -e "${RED}[错误]${NC} $*" >&2; }

# ---------- 默认配置 ----------
APP_NAME="tcm-tongue-diagnosis"
APP_PORT=7860
INSTALL_DIR="/opt/${APP_NAME}"
SERVICE_NAME="tcm-tongue"
RUN_USER="${SUDO_USER:-$(whoami)}"
AGNES_API_KEY=""
WITH_U2NET=false
WITH_NGINX=false
SKIP_LFS=false
REPO_URL="https://github.com/lhbzx1984/2026_AI_XLY.git"

# ---------- 参数解析 ----------
for arg in "$@"; do
    case "$arg" in
        --api-key=*)     AGNES_API_KEY="${arg#*=}";;
        --with-u2net)    WITH_U2NET=true;;
        --with-nginx)    WITH_NGINX=true;;
        --with-ml)       WITH_ML=true;;
        --skip-lfs)      SKIP_LFS=true;;
        --repo=*)        REPO_URL="${arg#*=}";;
        --port=*)        APP_PORT="${arg#*=}";;
        --help|-h)
            echo "用法: sudo bash deploy.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --api-key=KEY      指定 Agnes AI API Key（不指定则跳过 AI 评语）"
            echo "  --with-u2net       安装 U2-Net 分割后端（rembg + onnxruntime）"
            echo "  --with-ml          安装 PyTorch 深度学习依赖（体积大，约 2GB）"
            echo "  --with-nginx       配置 Nginx 反向代理（80 → ${APP_PORT}）"
            echo "  --skip-lfs         跳过 Git LFS 大文件下载（手动上传 models/u2net.onnx）"
            echo "  --repo=URL         指定 Git 仓库地址（默认：${REPO_URL}）"
            echo "  --port=PORT        指定应用端口（默认：${APP_PORT}）"
            echo "  --help             显示帮助"
            exit 0
            ;;
        *)
            log_warn "未知参数: $arg（已忽略）"
            ;;
    esac
done

# ---------- 前置检查 ----------
check_prerequisites() {
    log_info "检查运行环境..."

    # 检查是否 root
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要 root 权限，请使用 sudo 运行"
        exit 1
    fi

    # 检查操作系统
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        log_info "操作系统: ${PRETTY_NAME:-未知}"
        if [[ "${ID:-}" != "ubuntu" && "${ID:-}" != "debian" ]]; then
            log_warn "推荐 Ubuntu/Debian 系统，当前系统可能不完全兼容"
        fi
    else
        log_warn "无法识别操作系统类型"
    fi

    # 检查 Python 版本
    if command -v python3 &>/dev/null; then
        PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        log_info "Python 版本: ${PY_VERSION}"
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)'; then
            log_ok "Python 版本满足要求 (>=3.10)"
        else
            log_error "Python 版本过低 (当前 ${PY_VERSION}，需 >=3.10)"
            log_info "Ubuntu 22.04 自带 Python 3.10，建议升级系统或使用 Ubuntu 22.04+"
            exit 1
        fi
    else
        log_warn "未检测到 python3，将安装"
    fi

    log_ok "环境检查通过"
}

# ---------- 安装系统依赖 ----------
install_system_deps() {
    log_info "安装系统依赖（Python、Git、Nginx 等）..."

    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq \
        python3 python3-pip python3-venv \
        git git-lfs curl wget \
        libgl1 libglib2.0-0 \
        ca-certificates >/dev/null 2>&1

    # 初始化 Git LFS（用于 models/u2net.onnx 等大文件）
    git lfs install &>/dev/null || log_warn "git-lfs 安装或初始化失败，大文件可能无法正确拉取"

    if [[ "$WITH_NGINX" == true ]]; then
        apt-get install -y -qq nginx >/dev/null 2>&1
    fi

    log_ok "系统依赖安装完成"
}

# ---------- 安装 uv 包管理器 ----------
install_uv() {
    if command -v uv &>/dev/null; then
        log_info "uv 已安装: $(uv --version)"
        return
    fi

    log_info "安装 uv 包管理器..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    source "$HOME/.local/bin/env" 2>/dev/null || true

    # 加入 PATH（对所有用户生效）
    if ! grep -q '.local/bin/env' "$HOME/.bashrc"; then
        echo 'source $HOME/.local/bin/env' >> "$HOME/.bashrc"
    fi

    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &>/dev/null; then
        log_ok "uv 安装完成: $(uv --version)"
    else
        log_error "uv 安装失败"
        exit 1
    fi
}

# ---------- 获取项目代码 ----------
fetch_project() {
    if [[ -d "${INSTALL_DIR}/.git" ]]; then
        log_info "项目已存在，拉取最新代码..."
        cd "$INSTALL_DIR"
        git pull --ff-only || log_warn "git pull 失败，使用现有代码继续"
    else
        log_info "克隆项目到 ${INSTALL_DIR}..."
        rm -rf "$INSTALL_DIR"
        git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    # 拉取 Git LFS 大文件（如 models/u2net.onnx，约 176MB）
    if [[ "${SKIP_LFS:-false}" == true ]]; then
        log_warn "已跳过 LFS 大文件下载（--skip-lfs），请手动上传 models/u2net.onnx"
    else
        log_info "拉取 LFS 大文件（如 U2-Net 模型）..."
        git lfs pull || log_warn "git lfs pull 失败，U2-Net 本地模型可能缺失（rembg 会在首次使用时自动下载）"
    fi

    log_ok "项目代码就绪"

    # 修正文件归属（让普通用户可读写）
    chown -R "$RUN_USER":"$RUN_USER" "$INSTALL_DIR"
}

# ---------- 安装 Python 依赖 ----------
install_python_deps() {
    log_info "安装 Python 依赖（首次可能较慢）..."
    cd "$INSTALL_DIR"

    export PATH="$HOME/.local/bin:$PATH"

    # 主依赖
    sudo -u "$RUN_USER" env PATH="$HOME/.local/bin:$PATH" uv sync --no-dev

    # 可选：U2-Net 分割后端
    if [[ "$WITH_U2NET" == true ]]; then
        log_info "安装 U2-Net 分割后端（rembg + onnxruntime）..."
        sudo -u "$RUN_USER" env PATH="$HOME/.local/bin:$PATH" uv add rembg onnxruntime
    fi

    # 可选：深度学习模型依赖
    if [[ "${WITH_ML:-false}" == true ]]; then
        log_info "安装 PyTorch 深度学习依赖（体积较大，请耐心等待）..."
        sudo -u "$RUN_USER" env PATH="$HOME/.local/bin:$PATH" uv sync --extra ml
    fi

    log_ok "Python 依赖安装完成"
}

# ---------- 配置 Agnes API Key ----------
configure_api_key() {
    cd "$INSTALL_DIR"

    if [[ -z "$AGNES_API_KEY" ]]; then
        log_warn "未通过 --api-key 指定 Agnes API Key"
        log_info "AI 评语功能将不可用（可后续手动写入 ${INSTALL_DIR}/.agnes_api_key）"
        return
    fi

    log_info "配置 Agnes AI API Key..."
    echo -n "$AGNES_API_KEY" > "${INSTALL_DIR}/.agnes_api_key"
    chown "$RUN_USER":"$RUN_USER" "${INSTALL_DIR}/.agnes_api_key"
    chmod 600 "${INSTALL_DIR}/.agnes_api_key"
    log_ok "API Key 已配置（权限 600，仅所有者可读）"
}

# ---------- 创建 systemd 服务 ----------
create_systemd_service() {
    log_info "创建 systemd 服务: ${SERVICE_NAME}"

    UV_BIN="$HOME/.local/bin/uv"
    if [[ ! -x "$UV_BIN" ]]; then
        # 尝试其他常见路径
        UV_BIN=$(command -v uv 2>/dev/null || echo "/root/.local/bin/uv")
    fi

    cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=TCM Tongue Diagnosis AI App
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="GRADIO_SERVER_NAME=0.0.0.0"
Environment="GRADIO_SERVER_PORT=${APP_PORT}"
ExecStart=${UV_BIN} run python -m app.main
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME" >/dev/null 2>&1
    log_ok "systemd 服务已创建并设置开机自启"
}

# ---------- 配置 Nginx 反向代理 ----------
configure_nginx() {
    if [[ "$WITH_NGINX" != true ]]; then
        return
    fi

    log_info "配置 Nginx 反向代理（80 → ${APP_PORT}）..."

    cat > "/etc/nginx/sites-available/${SERVICE_NAME}" << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:PORT;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
EOF

    # 替换端口占位符
    sed -i "s/PORT/${APP_PORT}/g" "/etc/nginx/sites-available/${SERVICE_NAME}"

    # 启用站点
    ln -sf "/etc/nginx/sites-available/${SERVICE_NAME}" "/etc/nginx/sites-enabled/"
    rm -f /etc/nginx/sites-enabled/default

    # 测试并重启
    if nginx -t 2>/dev/null; then
        systemctl restart nginx
        systemctl enable nginx >/dev/null 2>&1
        log_ok "Nginx 反向代理配置完成（访问 http://服务器IP 即可）"
    else
        log_warn "Nginx 配置测试失败，请手动检查 /etc/nginx/sites-available/${SERVICE_NAME}"
    fi
}

# ---------- 启动应用 ----------
start_service() {
    log_info "启动应用服务..."
    systemctl restart "$SERVICE_NAME"

    # 等待服务启动（首次需加载 U2-Net 模型，可能较慢）
    for i in {1..60}; do
        if curl -s "http://127.0.0.1:${APP_PORT}" -o /dev/null 2>&1; then
            log_ok "应用已启动并响应"
            return
        fi
        sleep 2
    done

    log_warn "应用可能仍在启动中（首次需加载模型），查看日志："
    log_info "  sudo journalctl -u ${SERVICE_NAME} -f"
}

# ---------- 验证部署 ----------
verify_deployment() {
    log_info "验证部署..."

    # 服务状态
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_ok "服务运行中"
    else
        log_error "服务未正常运行，请检查日志"
        log_info "  sudo journalctl -u ${SERVICE_NAME} -n 50"
        return 1
    fi

    # 端口监听
    if ss -tlnp | grep -q ":${APP_PORT}"; then
        log_ok "端口 ${APP_PORT} 正在监听"
    else
        log_warn "端口 ${APP_PORT} 未监听"
    fi

    # HTTP 响应
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${APP_PORT}" | grep -q "200"; then
        log_ok "HTTP 响应正常（200）"
    fi

    # 获取公网 IP
    PUBLIC_IP=$(curl -s http://metadata.tencentyun.com/latest/meta-data/public-ipv4 2>/dev/null || \
                curl -s ifconfig.me 2>/dev/null || echo "服务器IP")

    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  ✅ 中医AI舌诊系统部署完成！${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "  访问地址："
    if [[ "$WITH_NGINX" == true ]]; then
        echo -e "    ${BLUE}http://${PUBLIC_IP}${NC}"
    else
        echo -e "    ${BLUE}http://${PUBLIC_IP}:${APP_PORT}${NC}"
    fi
    echo ""
    echo -e "  常用命令："
    echo -e "    查看状态：  sudo systemctl status ${SERVICE_NAME}"
    echo -e "    查看日志：  sudo journalctl -u ${SERVICE_NAME} -f"
    echo -e "    重启应用：  sudo systemctl restart ${SERVICE_NAME}"
    echo -e "    停止应用：  sudo systemctl stop ${SERVICE_NAME}"
    echo ""
    echo -e "  ${YELLOW}提醒：${NC}请在腾讯云控制台防火墙中放行端口 ${APP_PORT}（TCP）"
    if [[ "$WITH_NGINX" == true ]]; then
        echo -e "  ${YELLOW}提醒：${NC}已配置 Nginx，还需放行端口 80（TCP）"
    fi
    echo ""
    if [[ "$WITH_U2NET" == true ]]; then
        echo -e "  ${YELLOW}注意：${NC}U2-Net 模型通过 Git LFS 拉取（约 176MB），若 LFS 拉取失败将在首次使用时自动下载"
    fi
    echo ""
}

# ---------- 主流程 ----------
main() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  中医AI舌诊系统 - 一键部署（腾讯云 Ubuntu）${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    log_info "部署配置："
    log_info "  安装目录：  ${INSTALL_DIR}"
    log_info "  运行用户：  ${RUN_USER}"
    log_info "  应用端口：  ${APP_PORT}"
    log_info "  U2-Net：    ${WITH_U2NET}"
    log_info "  Nginx：     ${WITH_NGINX}"
    log_info "  AI 评语：   $([[ -n "$AGNES_API_KEY" ]] && echo '已配置' || echo '未配置')"
    echo ""

    check_prerequisites
    install_system_deps
    install_uv
    fetch_project
    install_python_deps
    configure_api_key
    create_systemd_service
    configure_nginx
    start_service
    verify_deployment
}

main "$@"
