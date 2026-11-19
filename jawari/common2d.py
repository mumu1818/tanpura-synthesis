"""Same helpers as common.py, but outputs of the 2-D scripts go to jawari/out/2d."""
import os

from common import (PKG, OUT_ROOT, GZ_MONO, analyze, report, spectrogram_png,  # noqa: F401
                    to_audio, metric_rows, mean_metric, smooth_track)

HERE = str(OUT_ROOT / "2d")
os.makedirs(HERE, exist_ok=True)
