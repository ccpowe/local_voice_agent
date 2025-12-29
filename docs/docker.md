# Docker 部署与排错指南

本指南用于在 Ubuntu + NVIDIA 环境下构建并运行本项目的 Docker 容器，同时记录常见问题与解决办法。

## 1) 前置条件
- Docker 已安装
- NVIDIA 驱动可用：`nvidia-smi` 能正常输出
- 如果需要 GPU 容器：安装 `nvidia-container-toolkit` 并配置 Docker 运行时

## 1.1) 镜像可运行条件（准确描述）
一个 Docker 镜像能否运行，主要取决于以下硬性条件：
1) CPU 架构匹配（x86_64 镜像只能在 x86_64 跑，ARM 同理）
2) 宿主机内核足够新且支持容器功能（cgroups、namespaces 等）
3) Docker 运行时版本足够新（否则某些语法/特性不可用）
4) 如需 GPU：宿主机需安装对应驱动 + `nvidia-container-toolkit`

发行版/系统版本本身不是决定性因素，但它决定了内核和 Docker 版本是否满足以上条件。

### 安装 NVIDIA 容器运行时（Ubuntu 22.04）
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 GPU：
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 2) 构建镜像
在项目根目录执行：
```bash
docker build -t local-voice-agent .
```

## 3) 运行容器（单容器）
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

说明：
- `--gpus all` 让容器使用 GPU
- `--device /dev/snd` 让容器访问声卡（仅 Linux）
- HF 缓存目录指向 `./data`，避免重复下载模型

## 4) Docker Compose 运行
创建 `.env`（可从 `.env.example` 复制）：
```
LVA_MODEL_NAME=your_model
LVA_API_KEY=your_key
LVA_BASE_URL=http://host.docker.internal:8317/v1
```

启动：
```bash
docker compose up --build
```

交互方式：
```bash
docker compose run --rm local-voice-agent
```

## 5) 常见问题与解决
### 5.1 构建失败：`Readme file does not exist`
原因：构建时未复制 `README.md`。  
解决：在 `uv sync` 之前先 `COPY README.md`。

### 5.2 构建失败：`Unable to determine which files to ship`
原因：`uv sync` 构建本地包时，`src/` 还未复制进镜像。  
解决：先 `COPY src ./src` 再执行 `uv sync`。

### 5.3 构建失败：`g++ failed: No such file or directory`
原因：依赖包需要 C++ 编译（如 `curated-tokenizers`）。  
解决：在镜像里安装 `g++`、`make`。

### 5.4 构建失败：`UV_HTTP_TIMEOUT` / 下载超时
原因：大包下载超时。  
解决：设置更长超时时间，例如 `UV_HTTP_TIMEOUT=300`。

### 5.5 容器无法访问宿主机服务
现象：`APIConnectionError` 或 `Connection error`。  
解决：
- Linux 下为 `host.docker.internal` 增加映射：
  ```yaml
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```
- 或直接用宿主机网关地址：`http://172.17.0.1:8317/v1`

### 5.6 CPU 上 `float16` 不支持
原因：CPU 不支持 `float16`。  
解决：自动降级为 `int8`，或确保 GPU 可见。

### 5.7 容器内 TTS 播放无声
现象：生成了 wav 文件，但容器内播放没有声音。  
原因：Linux 桌面通常使用 PulseAudio，容器默认走 ALSA 输出可能无声。  
解决：推荐将音频播放放到客户端（API 模式），或配置 PulseAudio 转发给容器。

## 6) 运行后的交互
启动后会出现提示：
```
请输入你的问题 (或输入's'进行语音输入):
```
你可以：
- 直接输入文字回车
- 输入 `s` / `speech` 进行语音输入
- 输入 `tts:on` / `tts:off` 开关语音输出
- 输入 `再见` 退出程序
