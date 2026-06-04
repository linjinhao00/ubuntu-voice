"""
Local transcript validation helpers.

These checks are deliberately conservative. They only block outputs that are
strongly associated with silence/low-confidence hallucination or model prompt
leakage; normal dictation text should pass through untouched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

from bytecli.service.audio_preprocess import AudioDiagnostics


HALLUCINATION_PATTERNS: tuple[str, ...] = (
    "请点准确认识英语和英语的语音转录",
    "请准确识别英语和英语的语音转录",
    "请准确识别语音",
    "谢谢观看",
    "感谢观看",
    "字幕由",
    "欢迎订阅",
    "请不吝点赞",
    "thanks for watching",
    "subscribe",
)


@dataclass(frozen=True)
class TranscriptValidation:
    text: str
    blocked: bool
    reason: str = ""


def validate_transcript(
    text: str,
    diagnostics: Optional[AudioDiagnostics] = None,
    extra_patterns: Iterable[str] = (),
) -> TranscriptValidation:
    cleaned = normalize_transcript(text)
    if not cleaned:
        return TranscriptValidation("", False)

    folded = _fold(cleaned)
    for pattern in (*HALLUCINATION_PATTERNS, *extra_patterns):
        if _fold(pattern) in folded:
            return TranscriptValidation("", True, "hallucination_pattern")

    if _has_pathological_repetition(cleaned):
        return TranscriptValidation("", True, "repetition")

    if diagnostics is not None:
        speech_ratio = diagnostics.speech_ratio
        if speech_ratio is not None and speech_ratio < 0.03 and len(cleaned) >= 8:
            return TranscriptValidation("", True, "low_vad_long_text")
        if diagnostics.rms < 0.001 and len(cleaned) >= 8:
            return TranscriptValidation("", True, "low_energy_long_text")

    return TranscriptValidation(cleaned, False)


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fold(text: str) -> str:
    normalized = normalize_transcript(text).lower()
    return re.sub(r"[\s，。！？、,.!?;:：；\"'“”‘’（）()\[\]{}<>《》-]+", "", normalized)


def _has_pathological_repetition(text: str) -> bool:
    folded = _fold(text)
    if len(folded) >= 8 and re.search(r"(.)\1{7,}", folded):
        return True
    for size in range(2, 7):
        if len(folded) >= size * 5:
            pattern = re.compile(r"(.{" + str(size) + r"})\1{4,}")
            if pattern.search(folded):
                return True
    words = normalize_transcript(text).split()
    if len(words) >= 6:
        for idx in range(len(words) - 5):
            if len(set(words[idx : idx + 6])) == 1:
                return True
    return False
