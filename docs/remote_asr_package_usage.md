# ByteCLI Remote ASR 并行包使用说明

生成日期：2026-06-05

## 安装包

新包路径：

```text
/home/linjh/byteCLI/bytecli-remote-asr_1.1.1-remote1_amd64.deb
```

备份路径：

```text
/home/linjh/byteCLI/release-backups/bytecli-remote-asr_1.1.1-remote1_20260605.deb
```

SHA256：

```text
b599b9192eec75db3ec10a9c771d8fcceef92a5a7ebdd7ee8029104b26aac76b
```

## 为什么不会和当前包冲突

该包是并行安装变体：

| 项目 | 当前包 | 远端变体包 |
| --- | --- | --- |
| Debian 包名 | `bytecli` | `bytecli-remote-asr` |
| 服务命令 | `/usr/bin/bytecli-service` | `/usr/bin/bytecli-remote-service` |
| 设置命令 | `/usr/bin/bytecli-settings` | `/usr/bin/bytecli-remote-settings` |
| systemd unit | `bytecli.service` | `bytecli-remote.service` |
| Python 代码 | `/usr/lib/python3/dist-packages/bytecli` | `/opt/bytecli-remote/lib/python3/dist-packages/bytecli` |
| 配置目录 | `~/.config/bytecli` | `~/.config/bytecli-remote` |
| 数据目录 | `~/.local/share/bytecli` | `~/.local/share/bytecli-remote` |

注意：两个版本仍然使用同一个 D-Bus 名称和同一个 F8 热键，因此不要同时运行。先停当前包，再启动远端变体包。

## 安装

```bash
cd /home/linjh/byteCLI
sudo dpkg -i bytecli-remote-asr_1.1.1-remote1_amd64.deb
```

这个包安装后不会自动启动。

## 配置 API Token

不要把 token 写进仓库。推荐写到远端变体的私有配置文件：

```bash
mkdir -p ~/.config/bytecli-remote
nano ~/.config/bytecli-remote/config.json
```

配置示例：

```json
{
  "model": "remote_glm_low_volume",
  "device": "gpu",
  "audio_input": "auto",
  "hotkey": {
    "keys": ["F8"]
  },
  "language": "zh",
  "auto_start": false,
  "history_max_entries": 50,
  "remote_asr": {
    "endpoint": "https://asr.linjh-personal.top/v1/audio/transcriptions",
    "api_token": "把家里服务器的 Bearer Token 填在这里",
    "timeout_seconds": 5.0,
    "fallback_model": "fun_asr_nano"
  }
}
```

`fallback_model` 当前是保留字段。远端请求失败时本版本会显示转录失败；需要兜底时，在设置面板手动切换到本地 `fun_asr_nano`。

也可以用环境变量覆盖：

```bash
export BYTECLI_REMOTE_ASR_TOKEN='你的 token'
```

## 启动远端变体

先停止当前版本：

```bash
systemctl --user stop bytecli.service
```

启动远端变体：

```bash
systemctl --user daemon-reload
systemctl --user start bytecli-remote.service
systemctl --user status bytecli-remote.service
```

打开远端变体设置面板：

```bash
bytecli-remote-settings
```

## 切回当前版本

```bash
systemctl --user stop bytecli-remote.service
systemctl --user start bytecli.service
```

## 当前支持的 4 个模型

远端变体的设置面板只显示 4 个模型：

| 模型 | Profile |
| --- | --- |
| 远端 GLM 低声 | `remote_glm_low_volume` |
| 远端 Qwen 1.7B | `remote_qwen_1_7b` |
| 远端 Fun-ASR-Nano | `remote_fun_asr_nano` |
| 本地 Fun-ASR-Nano | `fun_asr_nano` |

说明：ByteCLI 会随请求发送 `backend=glm_asr/qwen_asr/fun_asr`。如果家里服务器支持按请求选择 backend，这 3 个远端模型可直接在本地设置面板中切换；如果家里服务器只按 `.env` 常驻后端工作，则服务端会忽略请求里的 backend 字段，实际后端以家里服务器当前 `.env` 为准。

## 验证

查看服务日志：

```bash
journalctl --user -u bytecli-remote.service -n 100 --no-pager
```

确认远端服务可达：

```bash
curl -fsSL https://asr.linjh-personal.top/health
```

如果 F8 没有响应，先确认只运行了一个服务：

```bash
systemctl --user status bytecli.service
systemctl --user status bytecli-remote.service
```
