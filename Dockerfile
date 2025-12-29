FROM python:3.13-slim

ENV UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/model/hf \
    HF_HUB_CACHE=/app/data/model/hf \
    TRANSFORMERS_CACHE=/app/data/model/hf

WORKDIR /app

# 系统依赖：音频相关库 + C/C++ 扩展编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（Python 包管理器）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 先拷贝依赖清单，便于 Docker 缓存复用
COPY pyproject.toml uv.lock README.md ./
# 先拷贝源码，保证本地包可构建
COPY src ./src
# 安装运行时依赖
RUN uv sync --frozen --no-dev
# 预下载 spaCy 模型，避免运行时下载
RUN uv run python -m spacy download en_core_web_sm
# 可选：data 会在运行时被 volume 挂载覆盖
COPY data ./data

# 默认入口命令
CMD ["uv", "run", "voice_agent"]
