from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Any

import numpy as np

from shared.params import TanpuraParams
from shared.loss import compute_loss


# Type for any synthesizer function
SynthFn = Callable[[TanpuraParams, float], tuple[np.ndarray, int, float]]

# Default tunable parameter ranges
TUNE_RANGES: dict[str, dict[str, Any]] = {
    "bridge_height": {"low": -0.0008, "high": -0.0002, "step": 0.00005},
    "bridge_curve_p": {"low": 1.8, "high": 2.8, "step": 0.1},
    "K_contact": {"low": 1e5, "high": 5e6, "log": True},
    "sigma0": {"low": 0.1, "high": 0.4, "step": 0.05},
    "pluck_pos": {"low": 0.75, "high": 0.85, "step": 0.01},
    "jiva_pos": {"low": 0.1, "high": 0.6, "step": 0.01},
}


def run_optimization(
    synth_fn: SynthFn,
    ref_y: np.ndarray,
    sr: int,
    *,
    tune: list[str] | None = None,
    n_trials: int = 100,
    n_jobs: int = 1,
    duration: float = 1.5,
    db_path: str = "tanpura_cache.db",
    study_name: str = "tanpura_v1",
    seed: int = 42,
) -> dict[str, Any]:
    import optuna
    from optuna.samplers import TPESampler

    tune_set = set(tune or TUNE_RANGES.keys())

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=TPESampler(seed=seed),
    )

    def objective(trial):
        p = TanpuraParams()
        p.sr = sr
        p.n_modes = 80

        for name in tune_set:
            if name not in TUNE_RANGES:
                continue
            r = TUNE_RANGES[name]
            if r.get("log"):
                val = trial.suggest_float(name, r["low"], r["high"], log=True)
            else:
                val = trial.suggest_float(name, r["low"], r["high"], step=r.get("step"))
            setattr(p, name, val)

        try:
            audio, _, survival = synth_fn(p, duration)
            if survival < 1.0:
                return 1000.0 * (1.0 - survival) + 500.0
            return compute_loss(ref_y, audio, sr)
        except Exception:
            return 2000.0

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print(f"Best Loss: {study.best_value}")
    print(json.dumps(study.best_params, indent=2))

    best_p = TanpuraParams()
    for k, v in study.best_params.items():
        if hasattr(best_p, k):
            setattr(best_p, k, v)

    return {"params": best_p, "loss": study.best_value, "raw": study.best_params}
