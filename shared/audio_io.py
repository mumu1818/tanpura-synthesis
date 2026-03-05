from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa


def load_reference(path: str | Path, sr: int = 44100, duration: float | None = None) -> tuple[np.ndarray, int]:
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    if duration is not None:
        y = y[: int(duration * sr_out)]
    return y, sr_out


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    import scipy.io.wavfile
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak
    scipy.io.wavfile.write(str(path), sr, (audio * 32767).astype(np.int16))
