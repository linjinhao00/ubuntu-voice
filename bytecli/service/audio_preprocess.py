"""
Audio diagnostics and low-volume preprocessing for local ASR.

The service records 16 kHz mono float32 audio.  This module keeps the
preprocessing dependency-light: normalization is pure numpy, while VAD and
denoise backends are optional and fail open to energy-based checks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from bytecli.constants import AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)

TARGET_RMS_DBFS = -22.0
MAX_GAIN_DB = 24.0
PEAK_LIMIT = 0.89125094  # -1 dBFS
MIN_SPEECH_RATIO = 0.08
ENERGY_SILENCE_RMS = 0.0005


@dataclass
class AudioDiagnostics:
    samples: int
    duration_seconds: float
    rms: float
    peak: float
    rms_dbfs: float
    peak_dbfs: float
    clip_ratio: float
    speech_ratio: Optional[float] = None
    speech_backend: str = "energy"
    speech_detected: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PreprocessResult:
    audio: np.ndarray
    input: AudioDiagnostics
    output: AudioDiagnostics
    profile: str
    gain_db: float
    skipped: bool
    skip_reason: str = ""

    def to_metrics(self) -> dict[str, object]:
        return {
            "preprocess_profile": self.profile,
            "preprocess_gain_db": round(self.gain_db, 2),
            "input_rms_dbfs": round(self.input.rms_dbfs, 2),
            "input_peak_dbfs": round(self.input.peak_dbfs, 2),
            "output_rms_dbfs": round(self.output.rms_dbfs, 2),
            "output_peak_dbfs": round(self.output.peak_dbfs, 2),
            "speech_ratio": (
                round(self.output.speech_ratio, 4)
                if self.output.speech_ratio is not None
                else None
            ),
            "speech_backend": self.output.speech_backend,
            "speech_detected": self.output.speech_detected,
            "preprocess_skipped": self.skipped,
            "preprocess_skip_reason": self.skip_reason,
        }


class AudioPreprocessor:
    """Preprocess short dictation audio before ASR."""

    def __init__(self, profile: str = "vad_norm") -> None:
        self.profile = profile or "vad_norm"

    def process(self, audio_data: np.ndarray) -> PreprocessResult:
        audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        input_diag = diagnose_audio(audio)
        if audio.size == 0:
            return PreprocessResult(
                audio=audio,
                input=input_diag,
                output=input_diag,
                profile=self.profile,
                gain_db=0.0,
                skipped=True,
                skip_reason="empty",
            )

        profile = self.profile
        if profile == "raw":
            output_diag = diagnose_audio(audio, include_vad=True)
            skipped = not output_diag.speech_detected
            return PreprocessResult(
                audio=audio,
                input=input_diag,
                output=output_diag,
                profile=profile,
                gain_db=0.0,
                skipped=skipped,
                skip_reason="no_speech" if skipped else "",
            )

        processed = _remove_dc(audio)
        if profile == "denoise_norm":
            processed = _try_denoise(processed)

        speech_diag = diagnose_audio(processed, include_vad=True)
        if not speech_diag.speech_detected:
            original_speech_diag = diagnose_audio(audio, include_vad=True)
            if original_speech_diag.speech_detected:
                processed = audio
                speech_diag = original_speech_diag

        if not speech_diag.speech_detected:
            return PreprocessResult(
                audio=processed,
                input=input_diag,
                output=speech_diag,
                profile=profile,
                gain_db=0.0,
                skipped=True,
                skip_reason="no_speech",
            )

        gain = _normalization_gain(speech_diag)
        normalized = np.clip(processed * gain, -1.0, 1.0).astype(np.float32)
        output_diag = diagnose_audio(normalized, include_vad=True)
        return PreprocessResult(
            audio=normalized,
            input=input_diag,
            output=output_diag,
            profile=profile,
            gain_db=_linear_to_db(gain),
            skipped=False,
        )


def diagnose_audio(audio_data: np.ndarray, include_vad: bool = False) -> AudioDiagnostics:
    audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    samples = int(audio.size)
    duration = samples / float(AUDIO_SAMPLE_RATE)
    if samples == 0:
        return AudioDiagnostics(
            samples=0,
            duration_seconds=0.0,
            rms=0.0,
            peak=0.0,
            rms_dbfs=-120.0,
            peak_dbfs=-120.0,
            clip_ratio=0.0,
            speech_detected=False,
        )

    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    peak = float(np.max(np.abs(audio)))
    clip_ratio = float(np.mean(np.abs(audio) >= 0.999))
    speech_ratio: Optional[float] = None
    speech_backend = "energy"
    if include_vad:
        speech_ratio, speech_backend = detect_speech_ratio(audio)

    speech_detected = _is_speech(rms, speech_ratio)
    return AudioDiagnostics(
        samples=samples,
        duration_seconds=round(duration, 4),
        rms=rms,
        peak=peak,
        rms_dbfs=_dbfs(rms),
        peak_dbfs=_dbfs(peak),
        clip_ratio=clip_ratio,
        speech_ratio=speech_ratio,
        speech_backend=speech_backend,
        speech_detected=speech_detected,
    )


def detect_speech_ratio(audio_data: np.ndarray) -> tuple[Optional[float], str]:
    audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0, "empty"

    ratio = _silero_speech_ratio(audio)
    if ratio is not None:
        return ratio, "silero"

    ratio = _webrtc_speech_ratio(audio)
    if ratio is not None:
        return ratio, "webrtc"

    return None, "energy"


def should_skip_audio(audio_data: np.ndarray) -> bool:
    return AudioPreprocessor("vad_norm").process(audio_data).skipped


def _remove_dc(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    return (audio - float(np.mean(audio))).astype(np.float32)


def _normalization_gain(diag: AudioDiagnostics) -> float:
    if diag.rms <= 0.0:
        return 1.0
    desired = _db_to_linear(TARGET_RMS_DBFS - diag.rms_dbfs)
    gain = min(desired, _db_to_linear(MAX_GAIN_DB))
    if diag.peak > 0.0:
        gain = min(gain, PEAK_LIMIT / diag.peak)
    return max(gain, 1.0)


def _try_denoise(audio: np.ndarray) -> np.ndarray:
    try:
        import pyrnnoise  # type: ignore[import-not-found]
    except Exception:
        logger.debug("RNNoise requested but pyrnnoise is unavailable.")
        return audio

    try:
        denoiser = pyrnnoise.RNNoise()
        return np.asarray(denoiser.process_frame(audio), dtype=np.float32).reshape(-1)
    except Exception as exc:
        logger.debug("RNNoise preprocessing failed: %s", exc)
        return audio


def _silero_speech_ratio(audio: np.ndarray) -> Optional[float]:
    try:
        model, get_speech_timestamps = _load_silero()
    except Exception as exc:
        logger.debug("Silero VAD unavailable: %s", exc)
        return None

    try:
        timestamps = get_speech_timestamps(
            audio,
            model,
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_seconds=False,
        )
    except TypeError:
        timestamps = get_speech_timestamps(audio, model, return_seconds=False)
    except Exception as exc:
        logger.debug("Silero VAD failed: %s", exc)
        return None

    speech_samples = 0
    for item in timestamps:
        start = int(item.get("start", 0))
        end = int(item.get("end", 0))
        speech_samples += max(0, end - start)
    return min(1.0, speech_samples / max(1, int(audio.size)))


@lru_cache(maxsize=1)
def _load_silero():
    from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore[import-not-found]

    return load_silero_vad(), get_speech_timestamps


def _webrtc_speech_ratio(audio: np.ndarray) -> Optional[float]:
    try:
        import webrtcvad  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        vad = webrtcvad.Vad(1)
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2").tobytes()
        frame_ms = 30
        frame_bytes = int(AUDIO_SAMPLE_RATE * frame_ms / 1000) * 2
        if len(pcm16) < frame_bytes:
            return None
        voiced = 0
        total = 0
        for offset in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
            total += 1
            if vad.is_speech(pcm16[offset : offset + frame_bytes], AUDIO_SAMPLE_RATE):
                voiced += 1
        if total == 0:
            return None
        return voiced / total
    except Exception as exc:
        logger.debug("WebRTC VAD failed: %s", exc)
        return None


def _is_speech(rms: float, speech_ratio: Optional[float]) -> bool:
    if speech_ratio is not None:
        return speech_ratio >= MIN_SPEECH_RATIO
    return rms >= ENERGY_SILENCE_RMS


def _dbfs(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return max(-120.0, 20.0 * math.log10(value))


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _linear_to_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value)
