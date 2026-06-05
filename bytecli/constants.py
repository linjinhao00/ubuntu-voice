"""
ByteCLI shared constants.

Centralises all paths, D-Bus identifiers, timeouts, recording parameters,
design tokens, Whisper model metadata and default configuration values used
across the three ByteCLI processes.
"""

import os
import importlib.util

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
CONFIG_DIR: str = os.path.expanduser(
    os.environ.get(
        "BYTECLI_CONFIG_DIR",
        os.path.join(os.path.expanduser("~"), ".config", "bytecli"),
    )
)
DATA_DIR: str = os.path.expanduser(
    os.environ.get(
        "BYTECLI_DATA_DIR",
        os.path.join(os.path.expanduser("~"), ".local", "share", "bytecli"),
    )
)

CONFIG_FILE: str = os.path.join(CONFIG_DIR, "config.json")
HISTORY_FILE: str = os.path.join(DATA_DIR, "history.json")
LOG_FILE: str = os.path.join(DATA_DIR, "logs", "bytecli.log")
MODEL_DIR: str = os.path.join(DATA_DIR, "models")
EVAL_DIR: str = os.path.join(DATA_DIR, "eval")

_UID: str = str(os.getuid())
_RUNTIME_NAME: str = os.environ.get("BYTECLI_RUNTIME_NAME", "bytecli")
PID_FILE: str = os.path.join("/run", "user", _UID, f"{_RUNTIME_NAME}.pid")
INDICATOR_PID_FILE: str = os.path.join(
    "/run",
    "user",
    _UID,
    f"{_RUNTIME_NAME}-indicator.pid",
)

# ---------------------------------------------------------------------------
# D-Bus identifiers
# ---------------------------------------------------------------------------
DBUS_BUS_NAME: str = "com.bytecli.Service"
DBUS_OBJECT_PATH: str = "/com/bytecli/Service"
DBUS_INTERFACE: str = "com.bytecli.ServiceInterface"

# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------
START_TIMEOUT: int = 30
STOP_TIMEOUT: int = 10
MODEL_SWITCH_TIMEOUT: int = 300
RESTART_TIMEOUT: int = 40

# ---------------------------------------------------------------------------
# Recording parameters
# ---------------------------------------------------------------------------
MIN_RECORDING_DURATION: float = 0.3   # seconds – ignore shorter presses
MAX_RECORDING_DURATION: float = 300.0  # seconds – auto-stop ceiling
AUDIO_SAMPLE_RATE: int = 16000         # Hz (Whisper requirement)
AUDIO_CHANNELS: int = 1                # mono
AUDIO_BUFFER_FRAMES: int = 1024        # frames per callback

# ---------------------------------------------------------------------------
# Design token colours (dark theme)
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "background": "#111111",
    "foreground": "#FFFFFF",
    "card": "#1A1A1A",
    "muted": "#2E2E2E",
    "muted_foreground": "#B8B9B6",
    "primary": "#FF8400",
    "primary_foreground": "#111111",
    "border": "#2E2E2E",
    "secondary": "#2E2E2E",
    "secondary_foreground": "#FFFFFF",
    "success_foreground": "#B6FFCE",
    "error_foreground": "#FF5C33",
    "warning_foreground": "#FF8400",
    "info_foreground": "#B2B2FF",
    "input": "#2E2E2E",
    "destructive": "#FF5C33",
}

# ---------------------------------------------------------------------------
# Inference profile catalogue
# ---------------------------------------------------------------------------
def _dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _env_dir_available(env_name: str) -> bool:
    value = os.environ.get(env_name, "")
    return bool(value and os.path.isdir(os.path.expanduser(value)))


def _funasr_runtime_available() -> bool:
    return _dependency_available("funasr") and _dependency_available("torchaudio")


def _sherpa_runtime_available() -> bool:
    return _dependency_available("sherpa_onnx")


def _sherpa_sensevoice_model_dir() -> str:
    return os.path.expanduser(
        os.environ.get(
            "BYTECLI_SHERPA_SENSEVOICE_MODEL_DIR",
            os.path.join(
                MODEL_DIR,
                "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            ),
        )
    )


def _sherpa_funasr_nano_model_dir() -> str:
    return os.path.expanduser(
        os.environ.get(
            "BYTECLI_SHERPA_FUNASR_NANO_MODEL_DIR",
            os.path.join(
                MODEL_DIR,
                "sherpa-onnx-funasr-nano-int8-2025-12-30",
            ),
        )
    )


def _sherpa_sensevoice_available() -> bool:
    model_dir = _sherpa_sensevoice_model_dir()
    return (
        _sherpa_runtime_available()
        and os.path.isfile(os.path.join(model_dir, "tokens.txt"))
        and (
            os.path.isfile(os.path.join(model_dir, "model.int8.onnx"))
            or os.path.isfile(os.path.join(model_dir, "model.onnx"))
        )
    )


def _sherpa_funasr_nano_available() -> bool:
    model_dir = _sherpa_funasr_nano_model_dir()
    tokenizer_dirs = (
        os.path.join(model_dir, "Qwen3-0.6B"),
        os.path.join(model_dir, "Qwen3-0.6 B"),
    )
    return (
        _sherpa_runtime_available()
        and any(os.path.isdir(path) for path in tokenizer_dirs)
        and (
            os.path.isfile(os.path.join(model_dir, "encoder_adaptor.int8.onnx"))
            or os.path.isfile(os.path.join(model_dir, "encoder_adaptor.onnx"))
        )
        and (
            os.path.isfile(os.path.join(model_dir, "llm.int8.onnx"))
            or os.path.isfile(os.path.join(model_dir, "llm.fp16.onnx"))
            or os.path.isfile(os.path.join(model_dir, "llm.onnx"))
        )
        and (
            os.path.isfile(os.path.join(model_dir, "embedding.int8.onnx"))
            or os.path.isfile(os.path.join(model_dir, "embedding.onnx"))
        )
    )


_REMOTE_ASR_ENDPOINT = os.environ.get(
    "BYTECLI_REMOTE_ASR_ENDPOINT",
    "https://asr.linjh-personal.top/v1/audio/transcriptions",
)
_PROFILE_SET = os.environ.get("BYTECLI_PROFILE_SET", "").strip().lower()


INFERENCE_PROFILES: dict[str, dict[str, object]] = {
    "remote_glm_low_volume": {
        "display_name": "Remote GLM Low Volume",
        "description": "Home server GLM-ASR-Nano over HTTPS",
        "backend": "remote_asr",
        "remote_backend": "glm_asr",
        "model": "zai-org/GLM-ASR-Nano-2512",
        "endpoint": _REMOTE_ASR_ENDPOINT,
        "compute_type": "bfloat16",
        "visible": True,
    },
    "remote_qwen_1_7b": {
        "display_name": "Remote Qwen 1.7B",
        "description": "Home server Qwen3-ASR-1.7B over HTTPS",
        "backend": "remote_asr",
        "remote_backend": "qwen_asr",
        "model": "Qwen/Qwen3-ASR-1.7B",
        "endpoint": _REMOTE_ASR_ENDPOINT,
        "compute_type": "bfloat16",
        "visible": True,
    },
    "remote_fun_asr_nano": {
        "display_name": "Remote Fun-ASR Nano",
        "description": "Home server Fun-ASR-Nano over HTTPS",
        "backend": "remote_asr",
        "remote_backend": "fun_asr",
        "model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "endpoint": _REMOTE_ASR_ENDPOINT,
        "compute_type": "bfloat16",
        "visible": True,
    },
    "fast": {
        "display_name": "Fast",
        "description": "faster-whisper small int8_float16, beam=1",
        "backend": "faster_whisper",
        "model": "small",
        "compute_type": "int8_float16",
        "beam_size": 1,
        "best_of": 1,
        "condition_on_previous_text": False,
        "timestamps": False,
        "visible": True,
    },
    "balanced": {
        "display_name": "Balanced",
        "description": "faster-whisper small int8_float16, beam=3",
        "backend": "faster_whisper",
        "model": "small",
        "compute_type": "int8_float16",
        "beam_size": 3,
        "best_of": 1,
        "condition_on_previous_text": False,
        "timestamps": False,
        "visible": True,
    },
    "zh_fast": {
        "display_name": "SenseVoice Small",
        "description": "SenseVoiceSmall ONNX quantized",
        "backend": "sensevoice_onnx",
        "model": "SenseVoiceSmall",
        "compute_type": "quantized",
        "visible": (
            _dependency_available("funasr_onnx")
            and _env_dir_available("BYTECLI_SENSEVOICE_ONNX_MODEL_DIR")
        ),
    },
    "fun_asr_nano": {
        "display_name": "Fun-ASR Nano",
        "description": "Fun-ASR-Nano-2512 800M, Chinese/dialect focused",
        "backend": "funasr_nano",
        "model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "compute_type": "bfloat16",
        "language": "auto",
        "visible": _funasr_runtime_available(),
    },
    "sherpa_sensevoice": {
        "display_name": "Sherpa SenseVoice",
        "description": "sherpa-onnx SenseVoice Small int8, CPU-friendly",
        "backend": "sherpa_sensevoice",
        "model": _sherpa_sensevoice_model_dir(),
        "compute_type": "int8",
        "provider": "cpu",
        "num_threads": 2,
        "language": "auto",
        "use_itn": True,
        "visible": _sherpa_sensevoice_available(),
    },
    "sherpa_funasr_nano": {
        "display_name": "Sherpa FunASR Nano",
        "description": "sherpa-onnx Fun-ASR-Nano int8, CPU-friendly",
        "backend": "sherpa_funasr_nano",
        "model": _sherpa_funasr_nano_model_dir(),
        "compute_type": "int8",
        "provider": "cpu",
        "num_threads": 2,
        "language": "",
        "itn": True,
        "max_new_tokens": 256,
        "visible": _sherpa_funasr_nano_available(),
    },
    "experimental_qwen": {
        "display_name": "Qwen ASR 0.6B",
        "description": "Qwen3-ASR-0.6B transformers backend",
        "backend": "qwen_asr",
        "model": "Qwen/Qwen3-ASR-0.6B",
        "compute_type": "bfloat16",
        "max_inference_batch_size": 1,
        "max_new_tokens": 256,
        "visible": _dependency_available("qwen_asr"),
    },
    "glm_low_volume": {
        "display_name": "GLM Low Volume",
        "description": "GLM-ASR-Nano-2512 transformers backend",
        "backend": "glm_asr",
        "model": "zai-org/GLM-ASR-Nano-2512",
        "compute_type": "bfloat16",
        "max_new_tokens": 128,
        "visible": False,
    },
}

INFERENCE_PROFILE_ORDER: tuple[str, ...] = (
    "remote_glm_low_volume",
    "remote_qwen_1_7b",
    "remote_fun_asr_nano",
    "sherpa_sensevoice",
    "sherpa_funasr_nano",
    "fun_asr_nano",
    "zh_fast",
    "experimental_qwen",
    "fast",
    "balanced",
)

if _PROFILE_SET == "remote":
    INFERENCE_PROFILE_ORDER = (
        "sherpa_sensevoice",
        "sherpa_funasr_nano",
        "experimental_qwen",
        "fun_asr_nano",
    )
    for _profile_key, _profile in INFERENCE_PROFILES.items():
        _profile["visible"] = (
            _profile_key in INFERENCE_PROFILE_ORDER
            and bool(_profile.get("visible"))
        )

VISIBLE_INFERENCE_PROFILES: tuple[str, ...] = tuple(
    key
    for key in INFERENCE_PROFILE_ORDER
    if key in INFERENCE_PROFILES and INFERENCE_PROFILES[key].get("visible")
)

# Legacy OpenAI Whisper models are kept as explicit fallbacks and for
# compatibility with existing config files/tests. New defaults use profiles.
LEGACY_WHISPER_MODELS: dict[str, dict[str, str]] = {
    "tiny": {
        "display_name": "Fast (tiny)",
        "size": "~75 MB",
    },
    "small": {
        "display_name": "Balanced (small)",
        "size": "~465 MB",
    },
    "medium": {
        "display_name": "Accurate (medium)",
        "size": "~1.5 GB",
    },
}

_ALLOWED_PROFILE_KEYS = (
    set(INFERENCE_PROFILE_ORDER)
    if _PROFILE_SET == "remote"
    else set(INFERENCE_PROFILES)
)

WHISPER_MODELS: dict[str, dict[str, str]] = {
    **{
        key: {
            "display_name": str(value["display_name"]),
            "size": str(value["description"]),
        }
        for key, value in INFERENCE_PROFILES.items()
        if key in _ALLOWED_PROFILE_KEYS
    },
    **({} if _PROFILE_SET == "remote" else LEGACY_WHISPER_MODELS),
}

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "model": "experimental_qwen" if _PROFILE_SET == "remote" else "fast",
    "device": "gpu",
    "audio_input": "auto",
    "hotkey": {
        "keys": ["F8"],
    },
    "language": "en",
    "auto_start": False,
    "history_max_entries": 50,
    "remote_asr": {
        "endpoint": _REMOTE_ASR_ENDPOINT,
        "api_token": "",
        "timeout_seconds": 5.0,
        "fallback_model": "fun_asr_nano",
    },
    "text_correction": {
        "enabled": True,
        "backend": "qwen" if _PROFILE_SET == "remote" else "rules",
        "model": "Qwen/Qwen3-0.6B",
        "device": "auto",
        "max_chars": 120,
        "max_new_tokens": 80,
        "local_files_only": True,
        "min_free_vram_mb": 1200,
    },
}
