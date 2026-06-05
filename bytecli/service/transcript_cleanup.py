"""
Fast local transcript cleanup.

This module intentionally uses conservative rules instead of an LLM so it
does not add noticeable latency after ASR inference.
"""

from __future__ import annotations

import os
import json
import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger(__name__)
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

_CHINESE_PUNCT_TRANSLATION = str.maketrans(
    {
        ",": "，",
        "?": "？",
        "!": "！",
        ":": "：",
        ";": "；",
    }
)

_CONTEXTUAL_CORRECTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(非常|特别|很|有点|比较)?小的生意"), r"\1小的声音"),
    (re.compile(r"小声的生意"), "小声的声音"),
    (re.compile(r"把生意(调|放|开|关|变|弄)"), r"把声音\1"),
    (re.compile(r"重新再?打一个炮"), "重新打一个包"),
    (re.compile(r"重新再?打个炮"), "重新打个包"),
    (re.compile(r"打一个炮(?=[。！？；，、\s]|$)"), "打一个包"),
    (re.compile(r"打个炮(?=[。！？；，、\s]|$)"), "打个包"),
    (re.compile(r"准确认识(?=语音|中文|英文|英语|音频)"), "准确识别"),
)

_FILLER_START_RE = re.compile(
    r"^(?:[嗯呃额啊]+|那个|这个|就是|然后|怎么说呢|对吧|你知道吧)"
    r"(?:[，,、\s]+|(?=[。！？!?；;]|$))+"
)
_LEADING_HESITATION_RE = re.compile(r"^[嗯呃额啊]+(?=[\u4e00-\u9fff])")
_LEADING_PHRASE_FILLER_RE = re.compile(
    r"^(?:就是|然后)(?=(?:我|你|帮|给|把|再|重新|可以|需要|想|要|做|加|改))"
)
_FILLER_AFTER_PUNCT_RE = re.compile(
    r"([，。！？；、,.!?;\s])"
    r"(?:[嗯呃额啊]+|那个|这个|就是|然后|怎么说呢|对吧|你知道吧)"
    r"(?:[，,、\s]+|(?=[。！？!?；;]|$))"
)
_REPEATED_FILLER_RE = re.compile(r"(?:嗯|呃|额|啊){2,}")
_EXTRA_SPACE_RE = re.compile(r"\s+")
_SPACE_AROUND_CJK_PUNCT_RE = re.compile(r"\s*([，。！？；、])\s*")
_DUPLICATE_PUNCT_RE = re.compile(r"([，。！？；、])\1+")


@dataclass(frozen=True)
class CleanupResult:
    text: str
    changed: bool
    elapsed_ms: float
    backend: str = "rules"


@dataclass(frozen=True)
class TextCorrectionSettings:
    enabled: bool = True
    backend: str = "rules"
    model: str = "Qwen/Qwen3-0.6B"
    device: str = "auto"
    max_chars: int = 120
    max_new_tokens: int = 80
    local_files_only: bool = True
    min_free_vram_mb: int = 1200


def cleanup_enabled() -> bool:
    value = os.environ.get("BYTECLI_TEXT_CLEANUP", "1").strip().lower()
    return value not in _FALSE_VALUES


def cleanup_transcript(
    text: str,
    settings: Optional[TextCorrectionSettings] = None,
) -> CleanupResult:
    start = time.perf_counter()
    original = str(text or "")
    cleaned = unicodedata.normalize("NFKC", original).strip()
    if not cleaned:
        return CleanupResult("", original != "", _elapsed_ms(start))

    cleaned = _normalize_punctuation(cleaned)
    cleaned = _remove_fillers(cleaned)
    cleaned = _apply_contextual_corrections(cleaned)
    cleaned = _tidy_spacing(cleaned)

    settings = settings or load_text_correction_settings()
    if _should_use_qwen(settings, cleaned):
        qwen_text = _get_qwen_corrector().correct_if_ready(cleaned, settings)
        if qwen_text:
            qwen_text = _tidy_spacing(_normalize_punctuation(qwen_text))
            if _is_plausible_correction(cleaned, qwen_text):
                return CleanupResult(
                    qwen_text,
                    qwen_text != original,
                    _elapsed_ms(start),
                    "qwen",
                )

    return CleanupResult(cleaned, cleaned != original, _elapsed_ms(start))


def load_text_correction_settings() -> TextCorrectionSettings:
    data: dict[str, Any] = {}
    try:
        from bytecli.constants import CONFIG_FILE, DEFAULT_CONFIG

        data.update(DEFAULT_CONFIG.get("text_correction", {}))
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
                file_config = json.load(fh)
            if isinstance(file_config.get("text_correction"), dict):
                data.update(file_config["text_correction"])
        except (OSError, json.JSONDecodeError):
            pass
    except Exception:
        data = {}

    if os.environ.get("BYTECLI_TEXT_CORRECTION_BACKEND"):
        data["backend"] = os.environ["BYTECLI_TEXT_CORRECTION_BACKEND"]
    if os.environ.get("BYTECLI_TEXT_CORRECTION_MODEL"):
        data["model"] = os.environ["BYTECLI_TEXT_CORRECTION_MODEL"]
    if os.environ.get("BYTECLI_TEXT_CORRECTION_DEVICE"):
        data["device"] = os.environ["BYTECLI_TEXT_CORRECTION_DEVICE"]
    if os.environ.get("BYTECLI_TEXT_CORRECTION_LOCAL_FILES_ONLY"):
        data["local_files_only"] = (
            os.environ["BYTECLI_TEXT_CORRECTION_LOCAL_FILES_ONLY"].strip().lower()
            not in _FALSE_VALUES
        )

    return TextCorrectionSettings(
        enabled=bool(data.get("enabled", True)),
        backend=str(data.get("backend", "rules")),
        model=str(data.get("model", "Qwen/Qwen3-0.6B")),
        device=str(data.get("device", "auto")),
        max_chars=int(data.get("max_chars", 120)),
        max_new_tokens=int(data.get("max_new_tokens", 80)),
        local_files_only=bool(data.get("local_files_only", True)),
        min_free_vram_mb=int(data.get("min_free_vram_mb", 1200)),
    )


def warm_up_text_corrector(settings: Optional[TextCorrectionSettings] = None) -> None:
    settings = settings or load_text_correction_settings()
    if _should_enable_qwen(settings):
        _get_qwen_corrector().warm_up(settings)


def _normalize_punctuation(text: str) -> str:
    if _looks_chinese(text):
        text = text.translate(_CHINESE_PUNCT_TRANSLATION)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text


def _remove_fillers(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _FILLER_START_RE.sub("", text)
        text = _LEADING_HESITATION_RE.sub("", text)
        text = _LEADING_PHRASE_FILLER_RE.sub("", text)
        text = _FILLER_AFTER_PUNCT_RE.sub(r"\1", text)
    text = _REPEATED_FILLER_RE.sub("", text)
    return text


def _apply_contextual_corrections(text: str) -> str:
    for pattern, replacement in _CONTEXTUAL_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


def _tidy_spacing(text: str) -> str:
    text = _SPACE_AROUND_CJK_PUNCT_RE.sub(r"\1", text)
    text = _DUPLICATE_PUNCT_RE.sub(r"\1", text)
    text = _EXTRA_SPACE_RE.sub(" ", text).strip()
    text = re.sub(r"^[，,、。！？；;:\s]+", "", text)
    text = re.sub(r"[，,、；;:\s]+$", "", text)
    return text.strip()


def _should_use_qwen(settings: TextCorrectionSettings, text: str) -> bool:
    if not _should_enable_qwen(settings):
        return False
    if len(text) > settings.max_chars:
        return False
    return True


def _should_enable_qwen(settings: TextCorrectionSettings) -> bool:
    if not settings.enabled:
        return False
    if settings.backend.lower() != "qwen":
        return False
    if not settings.model:
        return False
    return True


def _is_plausible_correction(source: str, corrected: str) -> bool:
    if not corrected:
        return False
    if "\n" in corrected:
        return False
    if len(corrected) > max(len(source) * 2 + 20, 80):
        return False
    if any(marker in corrected.lower() for marker in ("assistant", "用户", "纠正后")):
        return False
    return True


class QwenTextCorrector:
    """Lazy Qwen3-0.6B text corrector.

    The model is loaded in the background. Calls to ``correct_if_ready`` never
    block waiting for model loading; they return ``None`` until the model is
    ready, which keeps dictation latency bounded.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._loading = False
        self._tokenizer = None
        self._model = None
        self._model_id = ""
        self._device = "cpu"
        self._error = ""
        self._load_epoch = 0

    def warm_up(self, settings: TextCorrectionSettings) -> None:
        with self._load_lock:
            if self._model is not None and self._model_id == settings.model:
                return
            if self._loading:
                return
            self._loading = True
            self._load_epoch += 1
            load_epoch = self._load_epoch

        thread = threading.Thread(
            target=self._load,
            args=(settings, load_epoch),
            daemon=True,
            name="qwen-text-corrector-load",
        )
        thread.start()

    def correct_if_ready(
        self,
        text: str,
        settings: TextCorrectionSettings,
    ) -> Optional[str]:
        if self._model is None or self._tokenizer is None:
            self.warm_up(settings)
            return None
        if self._model_id != settings.model:
            self.warm_up(settings)
            return None

        prompt = _build_qwen_prompt(text)
        with self._lock:
            try:
                import torch

                tokenizer = self._tokenizer
                model = self._model
                if hasattr(tokenizer, "apply_chat_template"):
                    try:
                        encoded = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                            return_tensors="pt",
                        )
                    except TypeError:
                        encoded = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        )
                else:
                    encoded = tokenizer(_plain_qwen_prompt(text), return_tensors="pt")

                if hasattr(encoded, "to"):
                    encoded = encoded.to(self._device)
                    input_len = encoded.shape[-1]
                    model_inputs = {
                        "input_ids": encoded,
                        "attention_mask": torch.ones_like(encoded),
                    }
                else:
                    model_inputs = {
                        key: value.to(self._device) if hasattr(value, "to") else value
                        for key, value in encoded.items()
                    }
                    if "attention_mask" not in model_inputs:
                        model_inputs["attention_mask"] = torch.ones_like(
                            model_inputs["input_ids"]
                        )
                    input_len = model_inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=settings.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    )
                new_tokens = outputs[0][input_len:]
                decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                return _sanitize_qwen_output(decoded)
            except Exception as exc:
                self._error = str(exc)
                _reclaim_torch_memory()
                return None

    def _load(self, settings: TextCorrectionSettings, load_epoch: int) -> None:
        model = None
        tokenizer = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = _resolve_qwen_device(torch, settings)
            if device is None:
                self._error = "insufficient_vram"
                logger.warning(
                    "Qwen text corrector skipped: insufficient free VRAM "
                    "(model=%s, min_free_vram_mb=%s).",
                    settings.model,
                    settings.min_free_vram_mb,
                )
                return

            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            model_ref = _resolve_local_model_ref(settings)
            logger.info(
                "Loading Qwen text corrector model=%s resolved=%s device=%s.",
                settings.model,
                model_ref,
                device,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=settings.local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=settings.local_files_only,
                dtype=dtype,
            )
            model.to(device)
            model.eval()
            with self._load_lock:
                if load_epoch != self._load_epoch:
                    logger.info("Discarding stale Qwen text corrector load.")
                    del model
                    del tokenizer
                    _reclaim_torch_memory()
                    return
            with self._lock:
                self._tokenizer = tokenizer
                self._model = model
                self._model_id = settings.model
                self._device = device
                self._error = ""
            logger.info("Qwen text corrector loaded successfully on %s.", device)
        except Exception as exc:
            del model
            del tokenizer
            self._error = str(exc)
            logger.warning("Qwen text corrector failed to load: %s", exc)
            _reclaim_torch_memory()
        finally:
            with self._load_lock:
                if load_epoch == self._load_epoch:
                    self._loading = False

    def unload(self) -> None:
        with self._load_lock:
            self._load_epoch += 1
            self._loading = False
        with self._lock:
            had_model = self._model is not None or self._tokenizer is not None
            self._tokenizer = None
            self._model = None
            self._model_id = ""
            self._device = "cpu"
            self._error = ""
        if had_model:
            logger.info("Qwen text corrector unloaded.")
        _reclaim_torch_memory()


def _resolve_qwen_device(torch_module, settings: TextCorrectionSettings) -> Optional[str]:
    requested = settings.device.lower()
    if requested in ("gpu", "cuda"):
        requested = "cuda"

    if requested == "cpu":
        return "cpu"
    if requested in ("auto", "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        free_bytes, _total_bytes = torch_module.cuda.mem_get_info()
        free_mb = free_bytes / (1024 * 1024)
        if free_mb >= settings.min_free_vram_mb:
            return "cuda:0"
        return None if requested == "auto" else "cpu"
    return "cpu" if requested != "auto" else None


def _resolve_local_model_ref(settings: TextCorrectionSettings) -> str:
    if not settings.local_files_only:
        return settings.model

    expanded = os.path.expanduser(settings.model)
    if os.path.isdir(expanded):
        return expanded

    repo_cache = "models--" + settings.model.replace("/", "--")
    snapshots_dir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        repo_cache,
        "snapshots",
    )
    if not os.path.isdir(snapshots_dir):
        return settings.model

    snapshots = [
        os.path.join(snapshots_dir, name)
        for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ]
    usable = [
        path
        for path in snapshots
        if os.path.exists(os.path.join(path, "config.json"))
        and (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "pytorch_model.bin"))
        )
    ]
    if not usable:
        return settings.model
    usable.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return usable[0]


def _build_qwen_prompt(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是语音转写文本纠错器。只修正明显的口癖、同音错字、标点和空格。"
                "必须保持原文语言：中文保持中文，不要翻译成英文；"
                "保留英文单词、缩写、代码词和命令词的英文形式与大小写，"
                "不要把 API、GPU、CUDA、Python、GitHub、Docker、SSH、HTTP、"
                "JSON、Qwen、FunASR、ByteCLI、Typeless 等技术词翻译成中文。"
                "不要改写语气，不要扩写，不要解释，只输出纠正后的文本。"
            ),
        },
        {"role": "user", "content": text},
    ]


def _plain_qwen_prompt(text: str) -> str:
    return (
        "只修正下面语音转写文本中的明显口癖、同音错字、标点和空格；"
        "必须保持原文语言，中文不要翻译成英文，英文不要翻译成中文；"
        "保留英文单词、缩写、代码词和命令词的英文形式与大小写；"
        "不要解释，只输出纠正后的文本：\n"
        f"{text}\n"
    )


def _sanitize_qwen_output(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^纠正后[:：]\s*", "", cleaned)
    cleaned = cleaned.strip("\"'“”‘’ \n\t")
    return cleaned


_CORRECTOR: Optional[QwenTextCorrector] = None
_CORRECTOR_LOCK = threading.Lock()


def _get_qwen_corrector() -> QwenTextCorrector:
    global _CORRECTOR
    with _CORRECTOR_LOCK:
        if _CORRECTOR is None:
            _CORRECTOR = QwenTextCorrector()
        return _CORRECTOR


def unload_text_corrector() -> None:
    with _CORRECTOR_LOCK:
        corrector = _CORRECTOR
    if corrector is not None:
        corrector.unload()


def _reclaim_torch_memory() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _looks_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
