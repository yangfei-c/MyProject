from __future__ import annotations

from pathlib import Path
from typing import Any
import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def check_audio(path: Path) -> Path:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio path is not a file: {audio_path}")
    return audio_path


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio_path = check_audio(path)
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def segment_audio(
    waveform: np.ndarray,
    sample_rate: int,
    segment_seconds: int,
    minimum_seconds: int,
) -> list[np.ndarray]:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if minimum_seconds <= 0:
        raise ValueError("minimum_seconds must be positive")

    waveform = np.asarray(waveform, dtype=np.float32)
    segment_samples = int(round(segment_seconds * sample_rate))
    minimum_samples = int(round(minimum_seconds * sample_rate))
    if segment_samples <= 0:
        raise ValueError("segment_seconds is too small for the given sample_rate")

    segments: list[np.ndarray] = []
    total_samples = int(waveform.shape[0])
    full_segment_count = total_samples // segment_samples

    for index in range(full_segment_count):
        start = index * segment_samples
        stop = start + segment_samples
        segments.append(np.asarray(waveform[start:stop], dtype=np.float32))

    remainder = waveform[full_segment_count * segment_samples :]
    if len(remainder) >= minimum_samples:
        segments.append(np.asarray(remainder, dtype=np.float32))

    return segments


def resample_audio(waveform: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if original_sample_rate == target_sample_rate:
        return waveform.astype(np.float32, copy=False)
    resampled = librosa.resample(
        y=waveform,
        orig_sr=original_sample_rate,
        target_sr=target_sample_rate,
    )
    return np.asarray(resampled, dtype=np.float32)


def build_audio_info(
    audio_path: Path,
    sample_rate: int,
    duration_seconds: float,
    segment_seconds: int,
    valid_segments: int,
) -> dict[str, Any]:
    return {
        "audio_path": str(Path(audio_path)),
        "file_name": Path(audio_path).name,
        "original_sample_rate": int(sample_rate),
        "duration_seconds": float(duration_seconds),
        "segment_seconds": int(segment_seconds),
        "valid_segments": int(valid_segments),
    }
