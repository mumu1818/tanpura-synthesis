# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "numba",
#     "scipy",
#     "librosa",
#     "optuna",
#     "cmaes",
# ]
# ///

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.params import TanpuraParams
from shared.audio_io import load_reference, write_wav
from shared.optimize import run_optimization
from waveguide.synthesizer import synthesize


def main():
    parser = argparse.ArgumentParser(description="Tanpura Waveguide Synthesis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    opt = sub.add_parser("optimize")
    opt.add_argument("--reference", required=True)
    opt.add_argument("--out", default="out_waveguide")
    opt.add_argument("--db", default="tanpura_cache.db")
    opt.add_argument("--study-name", default="waveguide_jiva_v1")
    opt.add_argument("--trials", type=int, default=100)
    opt.add_argument("--jobs", type=int, default=1)
    opt.add_argument("--duration", type=float, default=1.5)
    opt.add_argument("--seed", type=int, default=42)
    opt.add_argument(
        "--tune",
        nargs="*",
        default=["bridge_height", "bridge_curve_p", "K_contact", "sigma0", "pluck_pos", "jiva_pos"],
    )

    ren = sub.add_parser("render")
    ren.add_argument("--params", help="JSON file with params")
    ren.add_argument("--duration", type=float, default=10.0)
    ren.add_argument("--out", default="output_waveguide.wav")

    args = parser.parse_args()

    if args.cmd == "optimize":
        ref_y, sr = load_reference(args.reference, duration=args.duration)

        def synth_fn(p, dur):
            return synthesize(p, duration_s=dur)

        result = run_optimization(
            synth_fn, ref_y, sr,
            tune=args.tune,
            n_trials=args.trials,
            n_jobs=args.jobs,
            duration=args.duration,
            db_path=args.db,
            study_name=args.study_name,
            seed=args.seed,
        )

        out_dir = Path(args.out)
        out_dir.mkdir(exist_ok=True)
        (out_dir / "best_params.json").write_text(
            json.dumps(result["params"].to_dict(), indent=2)
        )

        print("Rendering best result...")
        audio, sr, _ = synthesize(result["params"], duration_s=10.0, verbose=True)
        write_wav(out_dir / "best_waveguide.wav", audio, sr)

    elif args.cmd == "render":
        p = TanpuraParams()
        if args.params:
            data = json.loads(Path(args.params).read_text())
            p = TanpuraParams.from_dict(data)
        audio, sr, _ = synthesize(p, duration_s=args.duration, verbose=True)
        write_wav(args.out, audio, sr)
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
