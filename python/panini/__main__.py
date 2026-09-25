"""Command line:  python -m panini <command> ...

  derive   "rāma+su gacchati"         full prakriyā with sūtra citations
  forms    "hare iha"                 every licensed surface form
  decline  rāma                       the implemented a-stem endings
  resolve  "rāma+bhis"                compare conflict-resolution theories of 1.4.2
  asiddha  "hare iha"                 with vs. without 8.2.1 pūrvatrāsiddham
  pratyahara ac | hal | …             expand a pratyāhāra
  compile                             compile vowel sandhi into a ReLU net and verify it
  speak    "rāmo gacchati" out.wav    deterministic speech synthesis
"""

from __future__ import annotations

import argparse
import sys

from . import prakriya, shivasutra, translit
from .engine import RESOLVERS, render_devanagari


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="panini", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("text")
    d.add_argument("--resolver", choices=RESOLVERS, default="classical")
    d.add_argument("--all", action="store_true", help="every choice vector")
    d.add_argument("--script", choices=("iast", "slp1", "deva"), default="iast")
    d.add_argument("--no-asiddha", action="store_true")
    f = sub.add_parser("forms")
    f.add_argument("text")
    f.add_argument("--resolver", choices=RESOLVERS, default="classical")
    sub.add_parser("decline").add_argument("stem")
    sub.add_parser("resolve").add_argument("text")
    sub.add_parser("asiddha").add_argument("text")
    p = sub.add_parser("pratyahara")
    p.add_argument("name")
    p.add_argument("--occurrence", type=int)
    sub.add_parser("compile")
    s = sub.add_parser("speak")
    s.add_argument("text")
    s.add_argument("out")
    a = ap.parse_args(argv)

    if a.cmd == "derive":
        ds = (prakriya.derive_all if a.all else lambda t, **k: [prakriya.derive(t, **k)])(
            a.text, resolver=a.resolver, asiddha=not a.no_asiddha)
        for i, dv in enumerate(ds):
            if a.all:
                print(f"── derivation {i + 1}/{len(ds)} ──")
            print(prakriya.format_trace(dv, a.script))
    elif a.cmd == "forms":
        for w in prakriya.forms(a.text, resolver=a.resolver):
            print(w, " ", render_devanagari(translit.iast_to_slp1(w)))
    elif a.cmd == "decline":
        names = {1: "prathamā", 2: "dvitīyā", 3: "tṛtīyā", 4: "caturthī", 5: "pañcamī"}
        for (vib, vac), fs in prakriya.decline(a.stem).items():
            print(f"{names[vib]:<9} {['eka', 'dvi', 'bahu'][vac - 1]:<5} {', '.join(fs)}")
    elif a.cmd == "resolve":
        for r in RESOLVERS:
            print(f"{r:<14} {', '.join(prakriya.forms(a.text, resolver=r))}")
    elif a.cmd == "asiddha":
        print("with 8.2.1   :", ", ".join(prakriya.forms(a.text)))
        print("without 8.2.1:", ", ".join(prakriya.forms(a.text, asiddha=False)))
    elif a.cmd == "pratyahara":
        try:                                   # SLP1 first ("Jal"), then IAST ("jhal")
            sounds = shivasutra.pratyahara(a.name, a.occurrence)
        except ValueError:
            sounds = shivasutra.pratyahara(translit.iast_to_slp1(a.name), a.occurrence)
        print(" ".join(translit.slp1_to_iast(x) for x in sounds))
    elif a.cmd == "compile":
        from .neural import compiled
        net = compiled.compile_ac_sandhi()
        ok, n, bad = compiled.verify(net)
        print(f"layers {net.mlp.shape}; {net.mlp.n_params} weights ({net.mlp.n_nonzero} non-zero)")
        print(f"agrees with the symbolic engine on {ok}/{n} vowel pairs")
        for x, y, want, got in bad:
            print("  mismatch", x, y, want, got)
        c = compiled.sivasutra_vs_random()
        print(f"threshold units for all {c['classes']} pratyāhāras: Śiva-sūtra order {c['sivasutra_units']}, "
              f"random orders {c['random_mean_units']:.0f} on average (best of 200: {c['random_min_units']})")
        return 0 if not bad else 1
    elif a.cmd == "speak":
        from .speech import synth
        audio = synth.speak(a.text, a.out)
        print(f"wrote {a.out} ({len(audio) / synth.SR:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
