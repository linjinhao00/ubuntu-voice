# ByteCLI ASR 优化验收报告

生成日期：2026-06-04
项目路径：`/home/linjh/byteCLI`
当前构建包：`bytecli_1.1.0_amd64.deb`
安装包备份：`release-backups/bytecli_1.1.0_amd64_20260604_asr-optimization.deb`
安装包 SHA256：`2b3dd066887be6358d0edd43d0c7e27c9ea5d438724a61b4a5775043a8d60118`

## 1. 验收结论

本轮已把 ByteCLI 从单一路径的 OpenAI Whisper 推理，整理为多后端 ASR Profile 架构。当前代码已支持 `faster-whisper`、`SenseVoiceSmall ONNX`、`Fun-ASR-Nano`、`Qwen3-ASR-0.6B`、`GLM-ASR-Nano` 和旧版 OpenAI Whisper fallback。

本机已经跑通并默认切换到 `fun_asr_nano`。它能在 RTX 2050 4GB 显存环境中加载并完成一次独立推理；系统服务也已能以 `fun_asr_nano` 启动。由于低声输入仍然是关键风险，最终准确率结论不能只看公开榜单，还需要用用户本机麦克风、真实小声输入、AISHELL6-Whisper 子集和静音样本继续做回归验收。

## 2. 发现并修复的问题

| 问题 | 修复结果 |
| --- | --- |
| Python 环境混乱，依赖容易装到 Anaconda 或错误解释器 | 安装脚本和 Debian postinst 改为固定使用 `/usr/bin/python3 -m pip`，降低服务运行环境与安装环境不一致的风险 |
| 默认 OpenAI Whisper 在 RTX 2050 4GB 上延迟和显存压力偏高 | 新增 Profile 架构，默认方向切到 `faster-whisper small int8_float16`，并保留 OpenAI Whisper fallback |
| 模型切换后 UI 可见项和实际可加载能力不一致 | `constants.py` 中按运行时依赖动态计算可见模型；SenseVoice ONNX 在缺少 `funasr_onnx` 或模型目录时不会误显示 |
| `fun_asr_nano` 依赖冲突和缓存路径不稳定 | 新增 FunASR Nano 加载路径、Hugging Face snapshot 本地缓存探测，以及 `torchaudio` 依赖检查 |
| Qwen 短音频容易输出固定幻觉文本 | 增加转录结果校验与幻觉拦截逻辑，避免把明显模板化输出直接插入 |
| 小声/弱音频与静音输入容易产生幻觉 | 增加 RMS 能量门限、音频预处理、静音短路和转录后校验 |
| F8 第二次按下不能及时停止录制 | 快捷键流程整理为 F8，并围绕录音状态机、D-Bus、hotkey manager 做了修复 |
| 配置面板过长，屏幕显示不下 | 配置窗口 UI 做了压缩布局和滚动/分区优化，模型选择区域也按可用项显示 |
| 包内容和最新代码不同步 | 已重新构建 Debian 包，生成 `/home/linjh/byteCLI/bytecli_1.1.0_amd64.deb` |

## 3. 当前支持的模型

| Profile | 后端 | 代码支持状态 | UI/默认策略 | 适用场景 |
| --- | --- | --- | --- | --- |
| `fast` | `faster-whisper small int8_float16` | 已支持 | 默认快速路径候选 | 速度优先、4GB 显存、本地短语音输入 |
| `balanced` | `faster-whisper small int8_float16 beam=3` | 已支持 | 可见 | 比 `fast` 稍慢，稳定性和准确率略优 |
| `zh_fast` | `SenseVoiceSmall ONNX quantized` | 已支持 | 仅当 `funasr_onnx` 和 `BYTECLI_SENSEVOICE_ONNX_MODEL_DIR` 可用时显示 | 中文短音频、低延迟 |
| `fun_asr_nano` | `FunAudioLLM/Fun-ASR-Nano-2512` | 已支持且本机跑通 | 当前配置已切到该模型 | 中文、方言、行业词、准确率优先但仍需实测 |
| `experimental_qwen` | `Qwen/Qwen3-ASR-0.6B` | 已支持 | 仅在 `qwen_asr` runtime 可用时显示 | 多语言、中文方言、实验验证 |
| `glm_low_volume` | `zai-org/GLM-ASR-Nano-2512` | 已支持 | 默认隐藏，仅离线评测 | 低声/耳语、小声输入准确率验证 |
| legacy `tiny/small/medium` | OpenAI Whisper PyTorch | 已保留 | fallback | 稳定兜底和兼容旧配置 |

## 4. 本地验收数据

| 项目 | 结果 |
| --- | --- |
| 单元/稳定性测试 | `/usr/bin/python3 -m pytest -q`：104 passed |
| 编译检查 | `/usr/bin/python3 -m compileall bytecli`：通过 |
| diff 空白检查 | `git diff --check`：通过 |
| 包构建 | `./scripts/build-deb.sh`：通过，生成 `bytecli_1.1.0_amd64.deb` |
| 安装包原始路径 | `/home/linjh/byteCLI/bytecli_1.1.0_amd64.deb` |
| 安装包备份路径 | `/home/linjh/byteCLI/release-backups/bytecli_1.1.0_amd64_20260604_asr-optimization.deb` |
| 安装包 SHA256 | `2b3dd066887be6358d0edd43d0c7e27c9ea5d438724a61b4a5775043a8d60118` |
| 服务启动 | `systemctl --user status bytecli`：active/running |
| 当前热键 | F8 |
| 当前本机配置模型 | `fun_asr_nano` |
| FunASR 本机加载 | `ASR profile 'fun_asr_nano' loaded successfully on 'gpu'.` |
| RTX 2050 4GB 显存占用 | FunASR 加载后约 2.27GB |

已做过一次独立推理烟测：`fun_asr_nano` 能加载并处理 1.5 秒测试音频。该音频是正弦波，不是人声，因此出现“嗯,对。”一类文本不能作为准确率通过，只能证明模型链路可执行。低声输入准确率仍需要真实语音样本验收。

## 5. 公开模型评测数据

这些数据来自模型官方仓库、模型卡或相关官方文档。不同模型的评测集不完全一致，因此只能用于排序和风险判断，不能替代本机麦克风验收。

| 模型 | 公开数据/官方说明 | 对 ByteCLI 的意义 |
| --- | --- | --- |
| GLM-ASR-Nano-2512 | 官方说明模型 1.5B 参数，针对 “Whisper/Quiet Speech” 训练，强调低声/小声鲁棒性，并称平均错误率 4.10，在 Wenet Meeting、AISHELL-1 等中文场景有优势 | 理论上最适合解决“小声输入听不清”，但 4GB 显存下不适合作为默认常驻模型，建议离线评测或服务器部署 |
| Fun-ASR-Nano-2512 | 模型卡说明 800M 参数，中文/英文/日文，中文支持 7 类方言和 26 类地域口音；训练数据为千万小时级真实语音；FunASR vLLM 文档中 PyTorch dynamic VAD 基线 CER 8.06%，vLLM batch CER 8.20%、RTFx 340 | 本机 4GB 可运行，是当前准确率和资源之间较现实的主力候选 |
| Qwen3-ASR-0.6B | 官方模型卡说明支持 30 种语言和 22 种中文方言；离线模式公开表中 Qwen3-ASR-0.6B 在 LibriSpeech、Fleurs-en、Fleurs-zh 平均 WER 为 3.48；内部集 Elders&Kids 4.48、ExtremeNoise 17.88、Dialog-Mandarin 7.06 | 多语言/方言能力强，但本机曾出现固定幻觉文本，需要作为实验模型并保留校验 |
| SenseVoiceSmall | 官方说明支持 50+ 语言，非自回归架构，10 秒音频约 70ms，约 15x Whisper-Large；同等参数量下比 Whisper-Small 快 5x 以上 | 低延迟很好，适合短语音；但当前 ONNX 模型目录未配置时不应显示 |
| faster-whisper small | faster-whisper 官方说明其基于 CTranslate2；CPU small benchmark 中 int8 RAM 1477MB，比 OpenAI Whisper fp32 2335MB 更省内存，速度也更快 | 是最稳的 4GB 显存快速 fallback，但中文小声准确率不是最优 |
| OpenAI Whisper small/medium | 旧路径保留为 fallback | 当新后端不可用或模型加载失败时兜底 |

## 6. 针对低声输入的专业验收建议

低声输入不能只靠换模型解决。建议用“固定样本 + 本机麦克风 + 多模型交叉验证”的方式验收：

1. 建立固定测试集：5 秒中文低声、10 秒中文夹英文技术词、15 秒 vibe coding 指令、5 秒静音、5 秒环境噪声、10 秒正常音量对照。
2. 加入 AISHELL6-Whisper 子集：该数据集包含约 30 小时中文耳语和 30 小时平行正常语音，适合做低声/耳语的最终模型评测。
3. 输出 CER/WER、总延迟、首字延迟、显存峰值、静音幻觉率、低声漏识别率。
4. 对每条音频跑至少两次，检查稳定性；短音频重复输出固定模板时判为幻觉。
5. 引入热词表：用户常用技术词、项目名、命令名和英文缩写应进入 hotwords 或 prompt。
6. 对小声输入启用前处理：自动增益、峰值限制、轻量降噪、RMS 门限和 VAD，不建议对低于 30 秒的短音频做复杂分段。

推荐验收排序：

| 目标 | 推荐顺序 |
| --- | --- |
| 准确率优先 | GLM-ASR-Nano 离线评测 > Fun-ASR-Nano > Qwen3-ASR-0.6B > SenseVoiceSmall > faster-whisper small |
| 4GB 显存现实优先 | SenseVoiceSmall ONNX / Fun-ASR-Nano > Qwen3-ASR-0.6B > faster-whisper small；GLM 只做离线评测或服务器 |
| 稳定兜底 | faster-whisper small 或 OpenAI Whisper small |

## 7. 优化建议

短期建议：

1. 把低声输入测试集固化到 `bytecli/eval`，让每次模型切换都能自动生成 CER/WER、耗时和显存报告。
2. 在配置面板显示“最近一次耗时、音频秒数、模型、显存峰值、是否拦截幻觉”，让用户知道当前模型是否适合。
3. 对 FunASR 增加 hotwords 配置入口，优先覆盖技术词、项目名和常用命令。
4. 对 Qwen 继续保留实验标签，只有通过本机低声样本后再开放为普通选项。
5. SenseVoice ONNX 应提供模型目录检测和安装提示，不要在依赖缺失时显示成可选模型。

中期建议：

1. 增加“本地模型”和“远端模型”两套 Profile，允许在局域网/服务器上跑大模型，本地只做录音和文本插入。
2. 做双模型校验：本地快速模型先给结果，低置信或疑似幻觉时异步请求高精度模型重写。
3. 增加轻量文本后处理：口语停顿修正、重复词去除、标点恢复、命令语气保留。
4. 将服务指标写入结构化日志，方便长期比较模型表现。

## 8. 对标 Typeless 的优缺点

Typeless 的定位更接近商业化 AI 语音键盘。App Store 信息显示它支持 iPhone，强调语音转清晰文本、滑动切回键盘、词建议和 36 种语言；官方指南还提到一次性语音校准、智能听写、翻译、Ask AI、个性化与隐私设置。

| 维度 | ByteCLI 优势 | ByteCLI 劣势 |
| --- | --- | --- |
| 平台 | Linux 桌面本地工具，可直接服务任意输入框 | 没有 Typeless 那样成熟的移动端键盘体验 |
| 隐私 | 可完全本地推理，语音不必上传 | 本地模型准确率受显卡和麦克风限制 |
| 可控性 | 可切换模型、调依赖、接服务器、看日志 | 对普通用户来说配置复杂 |
| 成本 | 本地运行，无订阅成本 | 需要本机硬件和环境维护 |
| 准确率 | 可以定制中文、低声、热词和服务器大模型 | 默认体验还没有商业产品的语音校准、云端大模型和后处理成熟 |
| 交互 | F8 热键适合桌面连续输入 | 缺少 swipe-to-type、词建议、移动端输入法联动 |

结论：ByteCLI 更适合“Linux 本地、隐私优先、可调模型”的开发者工作流；Typeless 更适合“开箱即用、移动端、云端智能润色”的普通用户场景。

## 9. 服务器/SSH 大模型方案评估

把模型部署在服务器上会提升准确率，尤其是可以运行本机 4GB 显存跑不稳的大模型，例如 GLM-ASR-Nano、Qwen3-ASR-1.7B、Whisper large-v3 或更强的服务化 ASR。但实现方式不应是每次录音都 `ssh` 启动一次命令，而应该是服务器常驻 ASR 服务，本地通过 SSH tunnel、HTTP 或 WebSocket 发送音频。

音频传输本身不是主要瓶颈。16kHz、16-bit、mono PCM 约 32KB/s，10 秒音频约 320KB；如果压缩成 WAV/FLAC/Opus，带宽压力更低。真正影响体验的是往返延迟、服务排队、模型推理时间和连接复用。

| 网络环境 | 体验判断 |
| --- | --- |
| 同机房/局域网，RTT 1-20ms | 很适合远端大模型，整体体验可能明显优于本地小模型 |
| 同城/稳定公网，RTT 20-80ms | 适合短语音输入，10 秒以内录音增加的网络延迟可接受 |
| 跨区域公网，RTT 100-300ms | 仍可用，但 F8 停止后的等待感明显，需要本地快速 fallback |
| 网络不稳定或服务器冷启动 | 不适合作为唯一方案，必须本地兜底 |

推荐架构：

1. 本地 ByteCLI 继续负责 F8、录音、静音检测、文本插入。
2. 服务器常驻 ASR HTTP/WebSocket 服务，模型常驻 GPU。
3. 用 SSH tunnel 保护链路，例如本地端口转发到服务器 ASR 服务。
4. 设定超时策略：例如 2.5 秒内远端未返回则先使用本地 FunASR/faster-whisper 结果。
5. 对低声输入优先走远端高精度模型；普通输入用本地模型。
6. 远端服务记录音频秒数、排队耗时、推理耗时和返回耗时。

综合判断：如果服务器 GPU 能稳定运行 GLM-ASR-Nano 或 Qwen3-ASR-1.7B，并且网络 RTT 小于 80ms，转录效果大概率会比本机 4GB 显存上的小模型更好；如果网络经常跨区或不稳定，本地模型仍应作为主路径或兜底路径。

## 10. 来源

- GLM-ASR 官方 GitHub：<https://github.com/zai-org/GLM-ASR>
- Fun-ASR-Nano Hugging Face：<https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512>
- FunASR vLLM Guide：<https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md>
- Qwen3-ASR-0.6B Hugging Face：<https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
- SenseVoice 官方 GitHub：<https://github.com/FunAudioLLM/SenseVoice>
- faster-whisper 官方 GitHub：<https://github.com/SYSTRAN/faster-whisper>
- AISHELL6-Whisper：<https://zutm.github.io/AISHELL6-Whisper/>
- Typeless App Store：<https://apps.apple.com/us/app/typeless-ai-voice-keyboard/id6749257650>
- Typeless User Guide：<https://typeless.cc/en/user-guide.html>
