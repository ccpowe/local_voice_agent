# voice_agent
本项目是多模态agent系统，能够进行asr语音识别加tts语音输出。
- 使用uv进行的环境管理

## 核心代码(src/local_voice_agent)
- voice_agent:agent系统
- speech_recognition:语音识别模块
- text_to_speech:语音合成模块

## 数据（data/）
- model:语音识别和生成模型
- voice:存放语音输出和临时语音文件
- images:存放图片文件

## 测试 （tests/）
- 存放测试代码

## Docker 部署
### 构建镜像
```bash
docker build -t local-voice-agent .
```

### 运行容器
```bash
docker run --rm -it \
  -e LVA_MODEL_NAME=your_model \
  -e LVA_API_KEY=your_key \
  -e LVA_BASE_URL=your_base_url \
  -e HF_HOME=/app/data/model/hf \
  -e HF_HUB_CACHE=/app/data/model/hf \
  -e TRANSFORMERS_CACHE=/app/data/model/hf \
  -v "$(pwd)/data:/app/data" \
  --gpus all \
  --device /dev/snd \
  local-voice-agent
```

### Docker Compose
在项目根目录创建 `.env`（可从 `.env.example` 复制），并设置：
```
LVA_MODEL_NAME=your_model
LVA_API_KEY=your_key
LVA_BASE_URL=
```

运行：
```bash
docker compose up --build
```

### 模型缓存持久化
为避免容器每次启动重复下载模型，已将 HuggingFace 缓存目录指向 `./data/model/hf`（通过 `HF_HOME`/`HF_HUB_CACHE`/`TRANSFORMERS_CACHE`）。

### 平台提示
Linux 可以通过 `--device /dev/snd` 直接访问声卡进行录音/播放。Mac/Windows 的 Docker 容器通常无法直接访问宿主机音频设备，适合作为纯推理/服务端场景使用。

### GPU 提示（Ubuntu + NVIDIA）
确保已安装 `nvidia-container-toolkit`，并可运行 `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` 验证容器可见 GPU。

## 常见问题与解决
### 1) Docker 构建失败：`Readme file does not exist`
原因：`pyproject.toml` 指定了 `readme = "README.md"`，但在构建依赖时未复制进镜像。  
解决：在 `uv sync` 之前先 `COPY README.md`。

### 2) Docker 构建失败：`Unable to determine which files to ship`
原因：`uv sync` 会构建本地包，但执行时还未复制 `src/`。  
解决：先 `COPY src ./src` 再执行 `uv sync`。

### 3) Docker 构建失败：`g++ failed: No such file or directory`
原因：依赖包需要编译 C++ 扩展（如 `curated-tokenizers`）。  
解决：在镜像里安装构建工具：`g++`、`make`。

### 4) Docker 构建失败：`UV_HTTP_TIMEOUT` / 下载超时
原因：大包下载超时（如 `nvidia-cufft-cu12`）。  
解决：设置更长超时时间，例如 `UV_HTTP_TIMEOUT=300`。

### 5) 容器内无法使用 GPU
现象：`docker run --gpus all ...` 报错或 GPU 仍不可用。  
解决：
1. 安装并配置 NVIDIA 容器运行时：
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
2. 验证：
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```
3. Compose 中使用 `gpus: all`，或 `docker run` 加 `--gpus all`。

### 6) Whisper 模型初始化失败：`float16` 不支持
原因：CPU 上不支持 `float16`。  
解决：在 CPU 环境下自动降级为 `int8`，或确保容器可见 GPU。

### 7) 容器内 TTS 播放无声
现象：生成了 wav 文件，但容器内播放没有声音。  
原因：Linux 桌面通常使用 PulseAudio，容器默认走 ALSA 输出可能无声。  
解决：推荐将音频播放放到客户端（API 模式），或配置 PulseAudio 转发给容器。

### 8) Whisper 运行时报错 `Unable to load any of {libcudnn_cnn.so...}`
原因：本地运行时 CUDA/cuDNN 动态库路径未加入 `LD_LIBRARY_PATH`，导致 GPU 推理失败。  
解决：将虚拟环境里的 cuDNN 路径加入环境变量，例如：
```bash
export LD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
```
