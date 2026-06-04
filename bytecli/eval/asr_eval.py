"""
Command-line ASR benchmark runner.

Manifest rows may be JSONL or CSV and must include at least:
audio_path, reference_text. Optional fields: scenario, speaker, source.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from bytecli.constants import AUDIO_SAMPLE_RATE, EVAL_DIR, INFERENCE_PROFILES
from bytecli.service.audio_preprocess import AudioPreprocessor, diagnose_audio
from bytecli.service.transcript_validation import validate_transcript
from bytecli.service.whisper_engine import WhisperEngine


DEFAULT_PROFILES = "glm_low_volume,fun_asr_nano,experimental_qwen,zh_fast,fast"
DEFAULT_PREPROCESSORS = "raw,vad_norm,denoise_norm"
DEFAULT_LEVELS = "0,-12,-24,-36"


@dataclass
class EvalSample:
    audio_path: str
    reference_text: str
    scenario: str = "default"
    speaker: str = ""
    source: str = ""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir or EVAL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_template:
        _write_template(Path(args.write_template))
        return 0

    if not args.manifest:
        print("--manifest is required unless --write-template is used.", file=sys.stderr)
        return 2

    samples = _load_manifest(Path(args.manifest))
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("Manifest contains no samples.", file=sys.stderr)
        return 2

    profiles = _split_csv(args.profiles)
    preprocessors = _split_csv(args.preprocessors)
    levels = [float(item) for item in _split_csv(args.levels)]
    repetitions = max(1, int(args.repetitions))

    rows: list[dict[str, object]] = []
    started = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = output_dir / f"asr-eval-{started}.jsonl"

    for profile in profiles:
        engine = WhisperEngine()
        try:
            engine.load_model(profile, args.device)
        except Exception as exc:
            rows.append(_load_error_row(profile, args.device, str(exc)))
            continue

        for preprocessor in preprocessors:
            engine.set_preprocessor_profile(preprocessor)
            for sample in samples:
                base_audio, sample_rate = _read_audio(Path(sample.audio_path))
                audio = _resample_if_needed(base_audio, sample_rate)
                for level_db in levels:
                    leveled = _apply_gain_db(audio, level_db)
                    for run_index in range(repetitions):
                        rows.append(
                            _run_one(
                                engine=engine,
                                profile=profile,
                                device=args.device,
                                preprocessor=preprocessor,
                                sample=sample,
                                audio=leveled,
                                level_db=level_db,
                                run_index=run_index,
                            )
                        )

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / f"asr-eval-{started}.csv"
    _write_csv(csv_path, rows)
    report_path = output_dir / f"asr-eval-{started}.md"
    _write_report(report_path, rows)

    print(f"Wrote JSONL: {jsonl_path}")
    print(f"Wrote CSV:   {csv_path}")
    print(f"Wrote report:{report_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local ByteCLI ASR evaluation.")
    parser.add_argument("--manifest", help="JSONL/CSV manifest with audio_path/reference_text.")
    parser.add_argument("--output-dir", help=f"Report directory. Default: {EVAL_DIR}")
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--preprocessors", default=DEFAULT_PREPROCESSORS)
    parser.add_argument("--levels", default=DEFAULT_LEVELS, help="Comma-separated gain dB variants.")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--write-template", help="Write a manifest template and exit.")
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> list[EvalSample]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

    samples: list[EvalSample] = []
    for row in rows:
        audio_path = str(row.get("audio_path", "")).strip()
        reference = str(row.get("reference_text", "")).strip()
        if not audio_path:
            continue
        samples.append(
            EvalSample(
                audio_path=audio_path,
                reference_text=reference,
                scenario=str(row.get("scenario", "default") or "default"),
                speaker=str(row.get("speaker", "") or ""),
                source=str(row.get("source", "") or ""),
            )
        )
    return samples


def _run_one(
    engine: WhisperEngine,
    profile: str,
    device: str,
    preprocessor: str,
    sample: EvalSample,
    audio: np.ndarray,
    level_db: float,
    run_index: int,
) -> dict[str, object]:
    started = time.perf_counter()
    raw_diag = diagnose_audio(audio, include_vad=False)
    try:
        text = engine.transcribe(audio)
        error = ""
    except Exception as exc:
        text = ""
        error = str(exc)
    elapsed = time.perf_counter() - started

    validation = validate_transcript(text)
    if validation.blocked:
        text = ""

    metrics = engine.last_metrics or {}
    reference = sample.reference_text
    row: dict[str, object] = {
        "profile": profile,
        "backend": INFERENCE_PROFILES.get(profile, {}).get("backend", ""),
        "device": device,
        "preprocessor": preprocessor,
        "audio_path": sample.audio_path,
        "scenario": sample.scenario,
        "speaker": sample.speaker,
        "source": sample.source,
        "level_db": level_db,
        "run_index": run_index,
        "reference_text": reference,
        "hypothesis_text": text,
        "cer": _cer(reference, text),
        "wer": _wer(reference, text),
        "chinese_cer": _cer(_chars_only(reference), _chars_only(text)),
        "elapsed_seconds": round(elapsed, 4),
        "rtf": round(elapsed / max(raw_diag.duration_seconds, 0.001), 4),
        "raw_rms_dbfs": round(raw_diag.rms_dbfs, 2),
        "raw_peak_dbfs": round(raw_diag.peak_dbfs, 2),
        "error": error,
        "validation_blocked": validation.blocked,
        "validation_reason": validation.reason,
    }
    row.update(metrics)
    return row


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore[import-not-found]

        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        audio = np.asarray(data, dtype=np.float32)
    except Exception:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())
            dtype = np.int16 if wav.getsampwidth() == 2 else np.uint8
            data = np.frombuffer(frames, dtype=dtype)
            if dtype is np.uint8:
                audio = (data.astype(np.float32) - 128.0) / 128.0
            else:
                audio = data.astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32).reshape(-1), int(sample_rate)


def _resample_if_needed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == AUDIO_SAMPLE_RATE or audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.size / float(sample_rate)
    target_size = max(1, int(round(duration * AUDIO_SAMPLE_RATE)))
    src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _apply_gain_db(audio: np.ndarray, db: float) -> np.ndarray:
    gain = 10.0 ** (db / 20.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def _cer(reference: str, hypothesis: str) -> float:
    ref = list(reference or "")
    hyp = list(hypothesis or "")
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    return round(_edit_distance(ref, hyp) / len(ref), 4)


def _wer(reference: str, hypothesis: str) -> float:
    try:
        import jiwer  # type: ignore[import-not-found]

        return round(float(jiwer.wer(reference, hypothesis)), 4)
    except Exception:
        ref = (reference or "").split()
        hyp = (hypothesis or "").split()
        if not ref:
            return 0.0 if not hyp else float(len(hyp))
        return round(_edit_distance(ref, hyp) / len(ref), 4)


def _chars_only(text: str) -> str:
    return "".join(ch for ch in str(text or "") if "\u4e00" <= ch <= "\u9fff")


def _edit_distance(reference: Iterable[str], hypothesis: Iterable[str]) -> int:
    ref = list(reference)
    hyp = list(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, 1):
        curr = [i]
        for j, hyp_item in enumerate(hyp, 1):
            cost = 0 if ref_item == hyp_item else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, object]]) -> None:
    successful = [row for row in rows if not row.get("error")]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in successful:
        key = (str(row.get("profile", "")), str(row.get("preprocessor", "")))
        groups.setdefault(key, []).append(row)

    lines = ["# ByteCLI ASR Eval Report", ""]
    lines.append(f"Rows: {len(rows)}")
    lines.append("")
    lines.append("| Profile | Preprocessor | CER | zh-CER | WER | P95 Latency | Failures |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (profile, preprocessor), group in sorted(groups.items()):
        latencies = [float(row.get("elapsed_seconds", 0.0)) for row in group]
        p95 = _percentile(latencies, 0.95)
        lines.append(
            "| {profile} | {preprocessor} | {cer:.4f} | {zh_cer:.4f} | "
            "{wer:.4f} | {p95:.3f}s | {failures} |".format(
                profile=profile,
                preprocessor=preprocessor,
                cer=_mean(row.get("cer") for row in group),
                zh_cer=_mean(row.get("chinese_cer") for row in group),
                wer=_mean(row.get("wer") for row in group),
                p95=p95,
                failures=sum(1 for row in group if row.get("validation_blocked")),
            )
        )

    error_rows = [row for row in rows if row.get("error")]
    if error_rows:
        lines.extend(["", "## Errors", ""])
        for row in error_rows[:20]:
            lines.append(
                "- {profile}/{preprocessor}: {error}".format(
                    profile=row.get("profile", ""),
                    preprocessor=row.get("preprocessor", ""),
                    error=row.get("error", ""),
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "audio_path": "/absolute/path/to/quiet_zh.wav",
            "reference_text": "我还能再搞一个就算是非常小的声音也能识别准确",
            "scenario": "quiet_zh",
            "speaker": "speaker01",
            "source": "local",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_error_row(profile: str, device: str, error: str) -> dict[str, object]:
    return {
        "profile": profile,
        "backend": INFERENCE_PROFILES.get(profile, {}).get("backend", ""),
        "device": device,
        "preprocessor": "",
        "audio_path": "",
        "scenario": "load_error",
        "error": error,
    }


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _mean(values: Iterable[object]) -> float:
    numbers = [float(value) for value in values if value not in (None, "")]
    return round(statistics.fmean(numbers), 4) if numbers else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return round(values[index], 4)


if __name__ == "__main__":
    raise SystemExit(main())
