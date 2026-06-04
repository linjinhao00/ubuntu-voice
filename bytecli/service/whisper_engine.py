"""
Speech-recognition engine wrapper.

The public class name stays ``WhisperEngine`` for compatibility with the
rest of the service, but the default runtime is now profile-driven:
``faster-whisper``/CTranslate2 for the fast paths, optional SenseVoice/Qwen
experimental backends, and OpenAI Whisper as a legacy fallback.
"""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import urllib.request
import wave
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import numpy as np

from bytecli.constants import (
    INFERENCE_PROFILES,
    LEGACY_WHISPER_MODELS,
    MODEL_DIR,
)
from bytecli.service.audio_preprocess import AudioPreprocessor, PreprocessResult
from bytecli.service.transcript_validation import validate_transcript

logger = logging.getLogger(__name__)

SHORT_AUDIO_SECONDS = 30.0
SILENCE_RMS_THRESHOLD = 0.003

# Whisper model download URLs (from openai/whisper source).
_WHISPER_MODEL_URLS: dict[str, str] = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
}

_WHISPER_MODEL_HASHES: dict[str, str] = {
    "tiny": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    "base": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
    "small": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    "medium": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
}

@dataclass
class TranscriptionMetrics:
    backend: str
    model: str
    profile: str
    compute_type: str
    audio_seconds: float
    inference_seconds: float
    total_seconds: float
    peak_vram_mb: Optional[float]
    preprocess_profile: str = ""
    preprocess_gain_db: float = 0.0
    input_rms_dbfs: Optional[float] = None
    input_peak_dbfs: Optional[float] = None
    output_rms_dbfs: Optional[float] = None
    output_peak_dbfs: Optional[float] = None
    speech_ratio: Optional[float] = None
    speech_backend: str = ""
    speech_detected: Optional[bool] = None
    hallucination_blocked: bool = False
    validation_reason: str = ""


class WhisperEngine:
    """Manages ASR runtime instances and thread-safe transcription."""

    def __init__(self) -> None:
        self._model = None
        self._loaded_models: dict[tuple[str, str], Any] = {}
        self._current_model: Optional[str] = None
        self._current_device: Optional[str] = None
        self._current_profile: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._loading = False
        self._last_metrics: Optional[TranscriptionMetrics] = None
        self._preprocessor_profile = os.environ.get(
            "BYTECLI_AUDIO_PREPROCESSOR", "vad_norm"
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    @property
    def current_device(self) -> Optional[str]:
        return self._current_device

    @property
    def last_metrics(self) -> Optional[dict[str, object]]:
        return asdict(self._last_metrics) if self._last_metrics else None

    def last_metrics_json(self) -> str:
        return json.dumps(self.last_metrics or {}, ensure_ascii=False)

    def set_preprocessor_profile(self, profile: str) -> None:
        self._preprocessor_profile = profile or "vad_norm"

    def load_model(self, model_name: str, device: str = "cpu") -> None:
        """Load or activate an inference profile/model on *device*."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        profile = self._resolve_profile(model_name)
        normalized_device = self._normalize_device(device)
        cache_key = (model_name, normalized_device)

        if cache_key in self._loaded_models:
            self._model = self._loaded_models[cache_key]
            self._current_model = model_name
            self._current_device = device
            self._current_profile = profile
            logger.info("Activated cached ASR profile '%s' on '%s'.", model_name, device)
            return

        backend = str(profile["backend"])
        logger.info(
            "Loading ASR profile '%s' backend=%s model=%s device=%s compute_type=%s ...",
            model_name,
            backend,
            profile.get("model"),
            normalized_device,
            profile.get("compute_type", ""),
        )

        try:
            if backend == "faster_whisper":
                model = self._load_faster_whisper(profile, normalized_device)
            elif backend == "openai_whisper":
                model = self._load_openai_whisper(profile, normalized_device)
            elif backend == "sensevoice_onnx":
                model = self._load_sensevoice_onnx(profile, normalized_device)
            elif backend == "funasr_nano":
                model = self._load_funasr_nano(profile, normalized_device)
            elif backend == "qwen_asr":
                model = self._load_qwen_asr(profile, normalized_device)
            elif backend == "glm_asr":
                model = self._load_glm_asr(profile, normalized_device)
            else:
                raise RuntimeError(f"Unsupported ASR backend '{backend}'.")
        except torch_cuda_oom_error():
            logger.error("Out of GPU memory while loading profile '%s'.", model_name)
            raise RuntimeError(f"GPU out of memory loading '{model_name}'.")
        except Exception as exc:
            logger.error("Failed to load ASR profile '%s': %s", model_name, exc)
            raise RuntimeError(f"Failed to load ASR profile '{model_name}': {exc}") from exc

        self._loaded_models[cache_key] = model
        self._model = model
        self._current_model = model_name
        self._current_device = device
        self._current_profile = profile
        logger.info("ASR profile '%s' loaded successfully on '%s'.", model_name, device)

    def _model_file_exists(self, model_name: str) -> bool:
        """Return True when no explicit first-run download is required."""
        profile = self._resolve_profile(model_name)
        if profile["backend"] != "openai_whisper":
            return True
        model_path = os.path.join(MODEL_DIR, f"{profile['model']}.pt")
        return os.path.isfile(model_path)

    def _download_model_file(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        profile = self._resolve_profile(model_name)
        if profile["backend"] != "openai_whisper":
            return

        runtime_model = str(profile["model"])
        if runtime_model not in _WHISPER_MODEL_URLS:
            return

        model_path = os.path.join(MODEL_DIR, f"{runtime_model}.pt")
        if os.path.isfile(model_path):
            logger.debug("Model file already exists: %s", model_path)
            return

        os.makedirs(MODEL_DIR, exist_ok=True)

        url = _WHISPER_MODEL_URLS[runtime_model]
        model_info = LEGACY_WHISPER_MODELS.get(runtime_model, {})
        size_str = model_info.get("size", "unknown size")

        if progress_callback:
            progress_callback(0, f"Downloading {runtime_model} model ({size_str})...")

        logger.info("Downloading OpenAI Whisper model '%s' from %s ...", runtime_model, url)

        tmp_path = model_path + ".part"
        try:
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024
            sha256 = hashlib.sha256()

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    sha256.update(chunk)
                    downloaded += len(chunk)

                    if total_size > 0 and progress_callback:
                        percent = min(int(downloaded * 100 / total_size), 99)
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        progress_callback(
                            percent,
                            f"Downloading... {mb_done:.0f}/{mb_total:.0f} MB",
                        )

            expected_hash = _WHISPER_MODEL_HASHES.get(runtime_model)
            if expected_hash and sha256.hexdigest() != expected_hash:
                os.remove(tmp_path)
                raise RuntimeError(
                    f"Model hash mismatch for '{runtime_model}'. Download may be corrupted."
                )

            os.rename(tmp_path, model_path)
            logger.info("Model '%s' downloaded to %s", runtime_model, model_path)

            if progress_callback:
                progress_callback(100, "Download complete. Loading model...")
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load_model_async(
        self,
        model_name: str,
        device: str = "cpu",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        done_callback: Optional[Callable[[bool, str], None]] = None,
    ) -> None:
        """Load the selected ASR profile in a background thread."""
        self._loading = True

        def _worker():
            try:
                if not self._model_file_exists(model_name):
                    self._download_model_file(model_name, progress_callback)
                elif progress_callback:
                    progress_callback(100, "Loading model...")

                self.load_model(model_name, device)
                self._loading = False
                if done_callback:
                    done_callback(True, "Model loaded successfully.")
            except Exception as exc:
                self._loading = False
                logger.error("Async ASR model load failed: %s", exc)
                if done_callback:
                    done_callback(False, str(exc))

        thread = threading.Thread(target=_worker, daemon=True, name="model-loader")
        thread.start()

    def unload_model(self) -> None:
        """Release all cached model instances and reclaim memory."""
        if self._model is None and not self._loaded_models:
            return

        logger.info("Unloading ASR models (%d cached).", len(self._loaded_models))
        self._loaded_models.clear()
        self._model = None
        self._current_model = None
        self._current_device = None
        self._current_profile = {}

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe(self, audio_data_np: np.ndarray) -> str:
        """Transcribe a float32 16 kHz numpy array to text."""
        if self._model is None:
            raise RuntimeError("No Whisper model is loaded.")

        audio = np.asarray(audio_data_np, dtype=np.float32).reshape(-1)
        preprocess = AudioPreprocessor(self._preprocessor_profile).process(audio)
        duration_s = len(preprocess.audio) / 16000.0
        if preprocess.skipped:
            logger.info(
                "Skipping audio before ASR (duration=%.2f s, reason=%s, "
                "input_rms=%.2f dBFS, speech_ratio=%s).",
                duration_s,
                preprocess.skip_reason,
                preprocess.input.rms_dbfs,
                preprocess.output.speech_ratio,
            )
            self._last_metrics = self._build_metrics(
                backend=str(self._current_profile.get("backend", "unknown")),
                duration_s=duration_s,
                inference_seconds=0.0,
                total_seconds=0.0,
                preprocess=preprocess,
            )
            return ""

        with self._lock:
            backend = str(self._current_profile.get("backend", "unknown"))
            total_start = time.perf_counter()
            self._reset_peak_vram()
            infer_start = time.perf_counter()
            try:
                if backend == "faster_whisper":
                    text = self._transcribe_faster_whisper(preprocess.audio, duration_s)
                elif backend == "openai_whisper":
                    text = self._transcribe_openai_whisper(preprocess.audio)
                elif backend == "sensevoice_onnx":
                    text = self._transcribe_sensevoice_onnx(preprocess.audio)
                elif backend == "funasr_nano":
                    text = self._transcribe_funasr_nano(preprocess.audio)
                elif backend == "qwen_asr":
                    text = self._transcribe_qwen_asr(preprocess.audio)
                elif backend == "glm_asr":
                    text = self._transcribe_glm_asr(preprocess.audio)
                else:
                    raise RuntimeError(f"Unsupported ASR backend '{backend}'.")
            except Exception as exc:
                logger.error("Transcription failed: %s", exc)
                raise RuntimeError(f"Transcription failed: {exc}") from exc
            finally:
                inference_seconds = time.perf_counter() - infer_start

            total_seconds = time.perf_counter() - total_start
            text = _collapse_repeats(text.strip())
            validation = validate_transcript(text, preprocess.output)
            text = validation.text
            self._last_metrics = self._build_metrics(
                backend=backend,
                duration_s=duration_s,
                inference_seconds=inference_seconds,
                total_seconds=total_seconds,
                preprocess=preprocess,
                hallucination_blocked=validation.blocked,
                validation_reason=validation.reason,
            )

        logger.info(
            "Transcription metrics: backend=%s model=%s compute_type=%s "
            "audio=%.2fs infer=%.2fs total=%.2fs peak_vram=%sMB",
            self._last_metrics.backend,
            self._last_metrics.model,
            self._last_metrics.compute_type,
            self._last_metrics.audio_seconds,
            self._last_metrics.inference_seconds,
            self._last_metrics.total_seconds,
            self._last_metrics.peak_vram_mb,
        )
        logger.debug("Transcription result: %r", text)
        return text

    def _build_metrics(
        self,
        backend: str,
        duration_s: float,
        inference_seconds: float,
        total_seconds: float,
        preprocess: PreprocessResult,
        hallucination_blocked: bool = False,
        validation_reason: str = "",
    ) -> TranscriptionMetrics:
        return TranscriptionMetrics(
            backend=backend,
            model=str(self._current_profile.get("model", self._current_model or "")),
            profile=self._current_model or "",
            compute_type=str(self._current_profile.get("compute_type", "")),
            audio_seconds=round(duration_s, 3),
            inference_seconds=round(inference_seconds, 3),
            total_seconds=round(total_seconds, 3),
            peak_vram_mb=self._peak_vram_mb(),
            preprocess_profile=preprocess.profile,
            preprocess_gain_db=round(preprocess.gain_db, 2),
            input_rms_dbfs=round(preprocess.input.rms_dbfs, 2),
            input_peak_dbfs=round(preprocess.input.peak_dbfs, 2),
            output_rms_dbfs=round(preprocess.output.rms_dbfs, 2),
            output_peak_dbfs=round(preprocess.output.peak_dbfs, 2),
            speech_ratio=(
                round(preprocess.output.speech_ratio, 4)
                if preprocess.output.speech_ratio is not None
                else None
            ),
            speech_backend=preprocess.output.speech_backend,
            speech_detected=preprocess.output.speech_detected,
            hallucination_blocked=hallucination_blocked,
            validation_reason=validation_reason,
        )

    @staticmethod
    def is_cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def is_probably_silent(audio_data_np: np.ndarray) -> bool:
        return AudioPreprocessor("vad_norm").process(audio_data_np).skipped

    def _resolve_profile(self, model_name: str) -> dict[str, Any]:
        if model_name in INFERENCE_PROFILES:
            return dict(INFERENCE_PROFILES[model_name])
        if model_name in LEGACY_WHISPER_MODELS:
            return {
                "backend": "openai_whisper",
                "model": model_name,
                "compute_type": "float16",
                "beam_size": 1,
                "best_of": 1,
                "condition_on_previous_text": False,
                "timestamps": False,
            }
        raise RuntimeError(f"Unknown ASR profile '{model_name}'.")

    def _normalize_device(self, device: str) -> str:
        if device == "gpu":
            return "cuda"
        return device

    def _load_faster_whisper(self, profile: dict[str, Any], device: str):
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install it with "
                "`/usr/bin/python3 -m pip install faster-whisper`."
            ) from exc

        compute_type = str(profile.get("compute_type", "int8_float16"))
        if device == "cpu" and compute_type == "int8_float16":
            compute_type = "int8"
            profile["compute_type"] = compute_type

        return WhisperModel(
            str(profile["model"]),
            device=device,
            compute_type=compute_type,
            download_root=MODEL_DIR,
        )

    def _load_openai_whisper(self, profile: dict[str, Any], device: str):
        try:
            import whisper  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("openai-whisper is not installed.") from exc

        return whisper.load_model(
            str(profile["model"]),
            device=device,
            download_root=MODEL_DIR,
        )

    def _load_sensevoice_onnx(self, profile: dict[str, Any], device: str):
        model_dir = os.environ.get("BYTECLI_SENSEVOICE_ONNX_MODEL_DIR")
        if not model_dir:
            raise RuntimeError(
                "SenseVoice ONNX requires BYTECLI_SENSEVOICE_ONNX_MODEL_DIR "
                "to point at a quantized SenseVoiceSmall ONNX model directory."
            )
        try:
            from funasr_onnx import SenseVoiceSmall  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "funasr-onnx is not installed. Install it before using zh_fast."
            ) from exc

        device_id = 0 if device == "cuda" else -1
        return SenseVoiceSmall(model_dir, batch_size=1, device_id=device_id)

    def _load_funasr_nano(self, profile: dict[str, Any], device: str):
        try:
            from funasr import AutoModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Fun-ASR-Nano requires funasr. Install it with "
                "`/usr/bin/python3 -m pip install --user funasr>=1.3.3`."
            ) from exc

        runtime_device = "cuda:0" if device == "cuda" else "cpu"
        explicit_model = os.environ.get("BYTECLI_FUN_ASR_MODEL_DIR")
        model_id = explicit_model or str(profile["model"])
        base_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device": runtime_device,
            "disable_update": True,
            "disable_pbar": True,
            "ncpu": int(os.environ.get("BYTECLI_FUN_ASR_NCPU", "2")),
        }
        hub = os.environ.get("BYTECLI_FUN_ASR_HUB", "hf")
        if hub:
            base_kwargs["hub"] = hub
        remote_code = os.environ.get("BYTECLI_FUN_ASR_REMOTE_CODE")
        if remote_code:
            base_kwargs["remote_code"] = remote_code
        vad_model = os.environ.get("BYTECLI_FUN_ASR_VAD_MODEL")
        if vad_model:
            base_kwargs["vad_model"] = vad_model
            base_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

        candidates = _funasr_model_candidates(model_id, include_cache=not explicit_model)
        last_error: Exception | None = None
        for candidate in candidates:
            kwargs = dict(base_kwargs)
            kwargs["model"] = candidate
            try:
                return AutoModel(**kwargs)
            except Exception as exc:
                last_error = exc
                logger.warning("Fun-ASR-Nano load failed with model=%s: %s", candidate, exc)
        raise RuntimeError(f"Fun-ASR-Nano failed to load: {last_error}")

    def _load_qwen_asr(self, profile: dict[str, Any], device: str):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Qwen3-ASR.") from exc

        try:
            from qwen_asr import Qwen3ASRModel  # type: ignore[import-untyped]
        except Exception as exc:
            raise RuntimeError(
                "qwen-asr could not be imported. Install or repair it with "
                '`/usr/bin/python3 -m pip install --user --upgrade '
                'qwen-asr>=0.0.6 Pillow`.'
            ) from exc

        model_id = os.environ.get("BYTECLI_QWEN_ASR_MODEL_DIR", str(profile["model"]))
        dtype = _torch_dtype(torch, str(profile.get("compute_type", "bfloat16")))
        device_map = "cuda:0" if device == "cuda" else "cpu"
        if device_map == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32

        return Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=int(profile.get("max_inference_batch_size", 1)),
            max_new_tokens=int(profile.get("max_new_tokens", 256)),
        )

    def _load_glm_asr(self, profile: dict[str, Any], device: str):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "GLM-ASR requires transformers from source and PyTorch. Install the "
                "experimental dependencies before using glm_low_volume."
            ) from exc

        model_id = os.environ.get("BYTECLI_GLM_ASR_MODEL_DIR", str(profile["model"]))
        dtype = _torch_dtype(torch, str(profile.get("compute_type", "bfloat16")))
        device_map = "cuda" if device == "cuda" else "cpu"
        if device_map == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        return {"processor": processor, "model": model, "device": device_map, "dtype": dtype}

    def _transcribe_faster_whisper(self, audio: np.ndarray, duration_s: float) -> str:
        vad_filter = duration_s > SHORT_AUDIO_SECONDS
        segments, _info = self._model.transcribe(
            audio,
            beam_size=int(self._current_profile.get("beam_size", 1)),
            best_of=int(self._current_profile.get("best_of", 1)),
            condition_on_previous_text=bool(
                self._current_profile.get("condition_on_previous_text", False)
            ),
            word_timestamps=False,
            vad_filter=vad_filter,
            compression_ratio_threshold=1.8,
            no_speech_threshold=0.6,
        )
        return "".join(segment.text for segment in segments).strip()

    def _transcribe_openai_whisper(self, audio: np.ndarray) -> str:
        use_fp16 = self._current_device != "cpu"
        result = self._model.transcribe(
            audio,
            fp16=use_fp16,
            condition_on_previous_text=False,
            compression_ratio_threshold=1.8,
            no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
        return str(result.get("text", "")).strip()

    def _transcribe_sensevoice_onnx(self, audio: np.ndarray) -> str:
        if hasattr(self._model, "generate"):
            result = self._model.generate(input=audio, language="auto", use_itn=True)
        else:
            result = self._model(audio)
        return _extract_text(result)

    def _transcribe_funasr_nano(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            _write_wav(tmp.name, audio)
            result = self._model.generate(
                input=[tmp.name],
                cache={},
                batch_size=1,
                language=str(self._current_profile.get("language", "中文")),
                itn=True,
                disable_pbar=True,
            )
        return _extract_text(result)

    def _transcribe_qwen_asr(self, audio: np.ndarray) -> str:
        kwargs = {
            "audio": (audio, 16000),
            "language": None,
            "return_time_stamps": False,
        }
        results = self._model.transcribe(**kwargs)
        if not results:
            return ""
        return _extract_text(results[0])

    def _transcribe_glm_asr(self, audio: np.ndarray) -> str:
        import torch

        processor = self._model["processor"]
        model = self._model["model"]
        device = self._model["device"]
        dtype = self._model["dtype"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            _write_wav(tmp.name, audio)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "url": tmp.name},
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

        if hasattr(inputs, "to"):
            inputs = inputs.to(device, dtype=dtype)
        else:
            inputs = {
                key: value.to(device, dtype=dtype) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(self._current_profile.get("max_new_tokens", 128)),
                do_sample=False,
            )
        input_ids = getattr(inputs, "input_ids", None)
        if input_ids is None and isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]
        decoded = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        return str(decoded[0]).strip() if decoded else ""

    def _reset_peak_vram(self) -> None:
        if self._normalize_device(self._current_device or "cpu") != "cuda":
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _peak_vram_mb(self) -> Optional[float]:
        if self._normalize_device(self._current_device or "cpu") != "cuda":
            return None
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        except Exception:
            pass
        return None


def _funasr_model_candidates(model_id: str, include_cache: bool = True) -> list[str]:
    expanded = os.path.expanduser(model_id)
    if os.path.isdir(expanded):
        return [expanded]

    candidates: list[str] = []
    if include_cache:
        cached = _find_huggingface_snapshot(model_id)
        if cached:
            candidates.append(cached)
    candidates.append(model_id)
    return _dedupe_preserving_order(candidates)


def _find_huggingface_snapshot(model_id: str) -> Optional[str]:
    repo_cache_name = "models--" + model_id.replace("/", "--")
    snapshots: list[str] = []
    for root in _huggingface_cache_roots():
        snapshot_glob = os.path.join(root, repo_cache_name, "snapshots", "*")
        snapshots.extend(path for path in glob.glob(snapshot_glob) if os.path.isdir(path))

    usable = [
        path
        for path in snapshots
        if os.path.isfile(os.path.join(path, "config.yaml"))
        and os.path.isfile(os.path.join(path, "model.pt"))
    ]
    if not usable:
        return None

    usable.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return usable[0]


def _huggingface_cache_roots() -> list[str]:
    roots: list[str] = []
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        roots.append(os.path.expanduser(hub_cache))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(os.path.join(os.path.expanduser(hf_home), "hub"))

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        roots.append(os.path.join(os.path.expanduser(xdg_cache), "huggingface", "hub"))

    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    return _dedupe_preserving_order(roots)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(result).strip()


def _torch_dtype(torch_module, dtype_name: str):
    normalized = dtype_name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch_module.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch_module.float16
    if normalized in ("fp32", "float32"):
        return torch_module.float32
    raise RuntimeError(f"Unsupported torch dtype '{dtype_name}'.")


def _collapse_repeats(text: str, max_repeat: int = 3) -> str:
    """Collapse runs of repeated characters or words."""
    import re

    text = re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1" * max_repeat, text)
    text = re.sub(
        r"\b(\w+)(?:\s+\1){" + str(max_repeat) + r",}",
        lambda m: (m.group(1) + " ") * max_repeat + m.group(1),
        text,
    )
    return text.strip()


def _write_wav(path: str, audio: np.ndarray) -> None:
    pcm = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm16.tobytes())


def torch_cuda_oom_error() -> type:
    """Return the CUDA OOM exception class, or a dummy that never matches."""
    try:
        import torch

        return torch.cuda.OutOfMemoryError
    except (ImportError, AttributeError):
        return type("_NeverRaised", (Exception,), {})
