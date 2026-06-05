# 家里电脑部署 `glm_low_volume` 远端 ASR 服务方案

生成日期：2026-06-04

目标：在家里 Windows 11 + WSL2 机器上部署 `zai-org/GLM-ASR-Nano-2512`，对外暴露 HTTPS ASR API，让 ByteCLI 本机通过公网域名调用该接口完成语音转录。

这份文档设计成可以直接交给家里电脑上的 Codex 执行。家里电脑当前没有 ByteCLI 仓库代码，因此本文档包含从零创建服务目录、安装环境、编写 API 服务、暴露公网接口、验收和 ByteCLI 对接所需的全部信息。

## 1. 当前家里电脑配置

硬件与系统：

| 项目 | 当前信息 |
| --- | --- |
| 宿主机 | Windows 11 家庭中文版 |
| WSL | Ubuntu 22.04.5 LTS |
| Kernel | `5.15.167.4-microsoft-standard-WSL2` |
| GPU | NVIDIA GeForce RTX 3070 |
| 显存 | 8192 MiB |
| 当前显存占用 | 约 2840 MiB / 8192 MiB |
| Driver | 591.86 |
| CUDA | `nvidia-smi` 显示 13.1 |
| CPU | AMD Ryzen 7 5800X，8 核 16 线程 |
| 宿主内存 | 约 32GB |
| WSL 可见内存 | 15Gi |
| WSL 根目录 | 1007G，总占用 41G，可用 916G |
| Python | 3.10.12 |
| pip | 23.1.2 |
| PyTorch | 未安装 |

判断：

- 这台机器适合做个人远端 ASR 服务器。
- RTX 3070 8GB 跑 GLM-ASR-Nano 属于边界可用，需要先释放显存并实测。
- 不建议一开始就把 GLM 作为唯一后端，应保留本地 fallback。
- 首次目标是跑通 `glm_low_volume` smoke test 和 HTTP API；如果 GLM 显存不稳，再切换家里服务器主力为 `Fun-ASR-Nano` 或 `Qwen3-ASR-0.6B`。

## 2. 总体架构

推荐架构：

```text
ByteCLI 本机
  F8 录音
  静音检测/音频预处理
  上传 WAV/FLAC
        |
        v
公网域名 HTTPS
  Caddy / Nginx 反向代理
  API Token 鉴权
        |
        v
家里电脑 WSL2
  FastAPI ASR 服务
  GLM-ASR-Nano 常驻 GPU
  返回 JSON 文本
```

不推荐架构：

```text
ByteCLI 每次录音后 ssh 到家里电脑执行 python transcribe.py
```

原因：SSH 建连、认证、进程启动和模型冷启动都会引入不可控延迟；常驻 HTTP 服务更适合输入法场景。

## 3. 官方依据

- GLM-ASR 官方 GitHub：<https://github.com/zai-org/GLM-ASR>
- GLM-ASR-Nano Hugging Face：<https://huggingface.co/zai-org/GLM-ASR-Nano-2512>
- NVIDIA CUDA on WSL：<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- PyTorch 官方安装版本页：<https://pytorch.org/get-started/previous-versions/>
- Caddy reverse_proxy 文档：<https://caddyserver.com/docs/caddyfile/directives/reverse_proxy>
- Caddy Automatic HTTPS 文档：<https://caddyserver.com/docs/automatic-https>

关键注意事项：

- GLM-ASR 官方说明模型针对低声/耳语场景训练，并提供 Transformers 推理示例。
- GLM-ASR Hugging Face 模型卡建议安装 `transformers` 源码版。
- NVIDIA WSL 文档说明 WSL2 使用 Windows 侧 NVIDIA 驱动，不要在 WSL 内安装 Linux 显卡驱动。
- Caddy 可做 HTTPS 和反向代理，但需要域名解析到家里公网 IP，并开放 80/443 或对应端口。

## 4. 先决条件

家里电脑需要满足：

1. Windows 上 `nvidia-smi` 正常显示 RTX 3070。
2. WSL2 Ubuntu 中也能运行 `nvidia-smi`。
3. WSL 不要休眠；Windows 不要自动睡眠。
4. 路由器可做端口转发，或已有内网穿透/反向代理方案。
5. 域名可解析到家里公网 IP，例如 `asr.linjh-personal.top`。
6. 家里宽带上行建议 >= 20Mbps；越高越好。
7. Windows 防火墙允许 Caddy 或代理服务入站。

建议先在 Windows PowerShell 管理员中设置 WSL 内存：

```powershell
notepad $env:USERPROFILE\.wslconfig
```

写入：

```ini
[wsl2]
memory=24GB
processors=12
swap=8GB
localhostForwarding=true
```

保存后重启 WSL：

```powershell
wsl --shutdown
wsl
```

说明：

- 你的宿主机有 32GB 内存，给 WSL 20-24GB 比较合理。
- `glm_low_volume` 可能因显存紧张失败，但内存不要再成为第二个瓶颈。

## 5. 创建服务目录

在家里电脑 WSL Ubuntu 中执行：

```bash
mkdir -p ~/asr-server
cd ~/asr-server
```

创建 Python 虚拟环境：

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv python3-pip ffmpeg git curl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 6. 安装 PyTorch 与依赖

优先使用 PyTorch CUDA 12.4 wheel。即使 `nvidia-smi` 显示 CUDA 13.1，PyTorch wheel 自带 CUDA runtime，通常不需要在 WSL 内单独安装完整 CUDA Toolkit。

```bash
source ~/asr-server/.venv/bin/activate
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("mem:", torch.cuda.mem_get_info() if torch.cuda.is_available() else None)
PY
```

预期：

- `cuda available: True`
- `gpu: NVIDIA GeForce RTX 3070`

如果这里失败：

1. 不要在 WSL 内安装 Linux NVIDIA driver。
2. 先升级 Windows NVIDIA 驱动。
3. 在 WSL 中确认 `/usr/lib/wsl/lib/libcuda.so` 存在。
4. 执行 `wsl --shutdown` 后重进 WSL。

安装 ASR 服务依赖：

```bash
pip install fastapi uvicorn[standard] python-multipart soundfile librosa numpy pydantic requests
pip install git+https://github.com/huggingface/transformers
pip install accelerate safetensors sentencepiece protobuf huggingface_hub
```

说明：

- `transformers` 用源码版是为了匹配 GLM-ASR 官方建议。
- `soundfile/librosa` 用于读取上传音频。
- `ffmpeg` 用于处理 mp3/m4a/webm 等非 WAV 输入。

## 7. 下载并加载 GLM-ASR-Nano

先设置 Hugging Face 缓存目录，建议放在 WSL 根目录或大盘挂载目录：

```bash
mkdir -p ~/asr-server/model-cache
export HF_HOME=~/asr-server/model-cache
export TRANSFORMERS_CACHE=~/asr-server/model-cache/transformers
```

如果在国内网络下载 Hugging Face 很慢，可以改用 ModelScope 或提前下载模型。第一版先用 Hugging Face，便于和官方示例一致。

创建 smoke test：

```bash
cat > ~/asr-server/smoke_glm.py <<'PY'
import os
import time
import torch
from transformers import AutoProcessor, AutoModel

os.environ.setdefault("HF_HOME", os.path.expanduser("~/asr-server/model-cache"))

repo_id = "zai-org/GLM-ASR-Nano-2512"

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("free/total:", torch.cuda.mem_get_info())

start = time.time()
processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
print("processor loaded in", round(time.time() - start, 2), "s")

start = time.time()
model = AutoModel.from_pretrained(
    repo_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model.eval()
print("model loaded in", round(time.time() - start, 2), "s")

if torch.cuda.is_available():
    print("after load free/total:", torch.cuda.mem_get_info())

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://github.com/zai-org/GLM-ASR/raw/main/examples/example_zh.wav",
            },
            {"type": "text", "text": "Please transcribe this audio into text"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device, dtype=torch.bfloat16)

start = time.time()
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
torch.cuda.synchronize() if torch.cuda.is_available() else None
print("inference seconds:", round(time.time() - start, 2))
print(processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True))
PY
```

运行：

```bash
cd ~/asr-server
source .venv/bin/activate
python smoke_glm.py
```

验收标准：

- 能加载模型。
- 不发生 CUDA OOM。
- 推理能输出中文文本。
- 加载后剩余显存最好 >= 1GB；如果小于 1GB，常驻服务风险较高。

如果 OOM：

1. 关闭 Windows 侧占 GPU 的程序，例如游戏、浏览器硬件加速、视频软件、AI 绘图工具。
2. 重启 WSL。
3. 再测 `nvidia-smi`，目标是空闲显存 >= 6.5GB。
4. 仍失败则不要强行部署 GLM，改用 `Fun-ASR-Nano` 或 `Qwen3-ASR-0.6B`。

## 8. 编写 FastAPI ASR 服务

创建服务文件：

```bash
cat > ~/asr-server/server.py <<'PY'
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from transformers import AutoModel, AutoProcessor

API_TOKEN = os.environ.get("ASR_API_TOKEN", "")
MODEL_ID = os.environ.get("ASR_MODEL_ID", "zai-org/GLM-ASR-Nano-2512")
MAX_UPLOAD_MB = int(os.environ.get("ASR_MAX_UPLOAD_MB", "20"))
MAX_NEW_TOKENS = int(os.environ.get("ASR_MAX_NEW_TOKENS", "256"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="Home GLM ASR Server", version="0.1.0")

processor = None
model = None
loaded_at = None


def check_auth(authorization: Optional[str]) -> None:
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def load_model_once() -> None:
    global processor, model, loaded_at
    if model is not None and processor is not None:
        return

    start = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=DTYPE,
        device_map=DEVICE,
    )
    model.eval()
    loaded_at = time.time()
    print(f"Loaded {MODEL_ID} on {DEVICE} in {loaded_at - start:.2f}s", flush=True)


@app.on_event("startup")
def startup() -> None:
    load_model_once()


@app.get("/health")
def health() -> dict:
    gpu = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "free_mb": round(free / 1024 / 1024),
            "total_mb": round(total / 1024 / 1024),
        }
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "loaded": model is not None,
        "loaded_at": loaded_at,
        "gpu": gpu,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
    response_format: str = "json",
):
    check_auth(authorization)
    load_model_once()

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    start_total = time.time()
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": tmp_path},
                    {"type": "text", "text": "Please transcribe this audio into text"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=model.dtype)

        start_infer = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.time() - start_infer

        text = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        total_seconds = time.time() - start_total
        result = {
            "text": text,
            "model": MODEL_ID,
            "backend": "glm_low_volume_remote",
            "inference_seconds": round(inference_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        }

        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse(result)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
PY
```

启动：

```bash
cd ~/asr-server
source .venv/bin/activate
export HF_HOME=~/asr-server/model-cache
export ASR_API_TOKEN='换成一个长随机字符串'
python server.py
```

本机 WSL 内测试：

```bash
curl -fsSL http://127.0.0.1:8008/health | python3 -m json.tool
```

上传音频测试：

```bash
curl -fsSL \
  -H "Authorization: Bearer $ASR_API_TOKEN" \
  -F "file=@/path/to/test.wav" \
  -F "response_format=json" \
  http://127.0.0.1:8008/v1/audio/transcriptions | python3 -m json.tool
```

## 9. systemd 用户服务

WSL 的 systemd 如果已启用，可以创建用户服务：

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/home-asr.service <<'EOF'
[Unit]
Description=Home GLM ASR Server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/asr-server
Environment=HF_HOME=%h/asr-server/model-cache
Environment=ASR_MODEL_ID=zai-org/GLM-ASR-Nano-2512
Environment=ASR_MAX_UPLOAD_MB=20
Environment=ASR_MAX_NEW_TOKENS=256
Environment=ASR_API_TOKEN=换成一个长随机字符串
ExecStart=%h/asr-server/.venv/bin/python %h/asr-server/server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
```

启动：

```bash
systemctl --user daemon-reload
systemctl --user enable --now home-asr.service
systemctl --user status home-asr.service
journalctl --user -u home-asr.service -f
```

如果 WSL 没启用 systemd，可在 `/etc/wsl.conf` 添加：

```ini
[boot]
systemd=true
```

然后 Windows PowerShell：

```powershell
wsl --shutdown
```

## 10. 公网域名暴露方案

推荐使用子域名：

```text
asr.linjh-personal.top
```

不建议复用：

```text
https://www.linjh-personal.top/project/
```

原因：ASR API 是独立服务，后续需要更清晰的路由、鉴权和日志；用子域名更干净。

### 10.1 路由器端口转发

如果家里有公网 IP：

1. 域名 DNS A 记录指向家里公网 IP。
2. 路由器转发 TCP 443 到家里电脑。
3. 可选转发 TCP 80 到家里电脑，用于 Caddy 自动签发证书和 HTTP 到 HTTPS 跳转。
4. Windows 防火墙允许 Caddy 入站。

如果家里公网 IP 会变：

- 配置 DDNS，自动更新 `asr.linjh-personal.top`。

如果没有公网 IP 或处于 CGNAT：

- 用 Cloudflare Tunnel、frp、Tailscale Funnel、ZeroTier 或反向 SSH 隧道。
- 这时不要尝试路由器端口转发。

### 10.2 Caddy 反向代理

Windows 侧或 WSL 侧都可以跑 Caddy。推荐先在 Windows 侧跑 Caddy，因为它更容易监听公网入站端口并和防火墙配合。

Caddyfile 示例：

```caddyfile
asr.linjh-personal.top {
    encode gzip

    @health path /health
    reverse_proxy @health 127.0.0.1:8008

    @asr path /v1/audio/transcriptions
    reverse_proxy @asr 127.0.0.1:8008

    respond "not found" 404
}
```

如果 Caddy 跑在 Windows，而 ASR 服务跑在 WSL，通常 `127.0.0.1:8008` 可通过 WSL localhost forwarding 访问。如果不通，改用 WSL IP：

```bash
hostname -I
```

然后 Caddyfile 中写：

```caddyfile
reverse_proxy 172.xx.xx.xx:8008
```

注意：WSL IP 可能变，优先用 localhost forwarding。

公网测试：

```bash
curl -fsSL https://asr.linjh-personal.top/health | python3 -m json.tool
```

## 11. 安全要求

必须做：

1. 使用 HTTPS。
2. 使用 `Authorization: Bearer <token>`。
3. 限制上传大小，当前服务默认 20MB。
4. 不把原始音频长期落盘，临时文件推理后删除。
5. 日志不要打印完整音频内容或敏感文本。
6. Windows 不要暴露 SSH 到公网。
7. 路由器只转发必要端口。

强烈建议：

1. 给 ASR 单独子域名。
2. API token 至少 32 字节随机字符串。
3. Caddy/Nginx 增加请求超时。
4. 如果 ByteCLI 本机公网出口固定，可限制来源 IP。
5. 定期更新 Windows NVIDIA 驱动、WSL 和 Python 依赖。

生成 token 示例：

```bash
python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
```

## 12. ByteCLI 本机后续对接要求

ByteCLI 当前仓库在 `/home/linjh/byteCLI`，后续需要新增一个远端 Profile。建议命名：

```text
remote_glm_low_volume
```

建议配置项：

```json
{
  "remote_asr": {
    "enabled": true,
    "profile": "remote_glm_low_volume",
    "endpoint": "https://asr.linjh-personal.top/v1/audio/transcriptions",
    "api_key": "不要写进仓库，放本地 config",
    "timeout_seconds": 5.0,
    "fallback_model": "fun_asr_nano"
  }
}
```

ByteCLI 调用协议：

```http
POST /v1/audio/transcriptions
Authorization: Bearer <token>
Content-Type: multipart/form-data

file=@recording.wav
response_format=json
```

期望响应：

```json
{
  "text": "转录文本",
  "model": "zai-org/GLM-ASR-Nano-2512",
  "backend": "glm_low_volume_remote",
  "inference_seconds": 1.234,
  "total_seconds": 1.456
}
```

ByteCLI 端行为：

1. F8 停止录音后先做本地 RMS 静音检测。
2. 非静音时上传 WAV。
3. 远端 5 秒内返回则使用远端文本。
4. 远端超时/报错/OOM/401/5xx 时自动 fallback 到本地 `fun_asr_nano`。
5. 记录远端耗时、HTTP 状态、fallback 原因。

## 13. 验收测试集

至少准备这些音频：

| 文件 | 内容 | 目标 |
| --- | --- | --- |
| `normal_zh_5s.wav` | 正常音量中文 5 秒 | 基础可用 |
| `quiet_zh_5s.wav` | 小声中文 5 秒 | 验证低声优势 |
| `mixed_zh_en_10s.wav` | 中文夹英文技术词 | 验证技术词 |
| `vibe_coding_15s.wav` | 编程指令 | 验证实际工作流 |
| `silence_5s.wav` | 静音 | 不应输出幻觉文本 |
| `noise_5s.wav` | 环境噪声 | 不应输出明显幻觉 |

验收指标：

| 指标 | 目标 |
| --- | --- |
| 服务启动 | 模型常驻成功，不 OOM |
| 5 秒普通语音 | 端到端等待尽量 < 3 秒 |
| 5 秒低声语音 | 明显优于本地 `fun_asr_nano` 或 `faster-whisper small` |
| 静音 | 返回空或低置信，不输出模板幻觉 |
| 稳定性 | 连续 20 次请求无崩溃 |
| 显存 | 推理后仍有余量，不持续增长 |

## 14. 性能排查命令

查看服务状态：

```bash
systemctl --user status home-asr.service
journalctl --user -u home-asr.service -n 100 --no-pager
```

查看 GPU：

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

测公网接口耗时：

```bash
curl -fsSL -o /dev/null \
  -w 'connect=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer} total=%{time_total} code=%{http_code}\n' \
  https://asr.linjh-personal.top/health
```

上传测试耗时：

```bash
time curl -fsSL \
  -H "Authorization: Bearer $ASR_API_TOKEN" \
  -F "file=@quiet_zh_5s.wav" \
  https://asr.linjh-personal.top/v1/audio/transcriptions
```

## 15. 常见失败与处理

### 15.1 CUDA OOM

处理：

1. 关闭 Windows 侧 GPU 程序。
2. 关闭浏览器硬件加速或视频播放。
3. 重启 WSL：`wsl --shutdown`。
4. 降低 `ASR_MAX_NEW_TOKENS=128`。
5. 改成 `Fun-ASR-Nano` 或 `Qwen3-ASR-0.6B`。

### 15.2 `torch.cuda.is_available()` 是 False

处理：

1. Windows 更新 NVIDIA 驱动。
2. WSL 中确认 `nvidia-smi` 可运行。
3. 不要在 WSL 内安装 Linux NVIDIA driver。
4. 重新安装 PyTorch CUDA wheel。

### 15.3 Hugging Face 下载慢

处理：

1. 使用代理。
2. 使用 `huggingface-cli download` 预下载。
3. 改用 ModelScope 下载到本地目录，再修改 `ASR_MODEL_ID` 为本地路径。

### 15.4 公网 HTTPS 不通

处理：

1. DNS A 记录是否指向当前公网 IP。
2. 路由器是否转发 80/443。
3. Windows 防火墙是否放行。
4. Caddy 是否启动。
5. WSL 服务是否监听 `0.0.0.0:8008`。

### 15.5 首次请求很慢

原因：

- 模型首次加载慢。
- 第一次 CUDA kernel 初始化慢。

处理：

- 服务启动时预加载模型。
- 启动后自动跑一次短音频 warmup。

## 16. 推荐执行顺序

家里电脑 Codex 应按这个顺序做：

1. 调整 WSL 内存到 20-24GB。
2. 安装 Python venv、ffmpeg、PyTorch CUDA wheel。
3. 验证 `torch.cuda.is_available()`。
4. 安装 Transformers 源码版和服务依赖。
5. 运行 `smoke_glm.py`。
6. 如果 GLM 加载成功，创建 `server.py`。
7. 本地 `curl http://127.0.0.1:8008/health`。
8. 本地上传 WAV 测试。
9. 配置 systemd 用户服务。
10. 配置 Caddy + 域名 + HTTPS。
11. 公网 `/health` 测试。
12. 公网上传 WAV 测试。
13. 把 endpoint、token、测试数据发回 ByteCLI 本机。
14. ByteCLI 本机实现 `remote_glm_low_volume` profile 和 fallback。

## 17. 是否继续使用 GLM 的决策门槛

满足以下条件才把 GLM 作为家里服务器的主力模型：

1. 能常驻加载，不 OOM。
2. 5 秒低声输入端到端 < 4 秒。
3. 低声准确率明显优于本地 `fun_asr_nano`。
4. 静音不输出幻觉。
5. 连续 20 次请求无显存增长和服务崩溃。

如果不满足，推荐家里服务器先部署：

```text
Fun-ASR-Nano > Qwen3-ASR-0.6B > faster-whisper medium/int8
```

GLM 保留为离线评测或手动高精度模式。
