"""Deterministic Sanskrit speech from Pāṇinian phonetics.

Every acoustic parameter is *derived*, not learned:

* **duration** — 1.2.27 ūkālo 'j jhrasvadīrghaplutaḥ: a vowel lasts one, two or
  three mātrās (the time of u, ū, u3); a consonant half a mātrā (Prātiśākhya).
* **pitch** — 1.2.29 uccair udāttaḥ (high), 1.2.30 nīcair anudāttaḥ (low),
  1.2.31 samāhāraḥ svaritaḥ (a combination), 1.2.32 tasyādita udāttam
  ardhahrasvam (the first half-mātrā of a svarita is high, then it falls).
* **spectrum** — from the sthāna/prayatna table in `panini.varna`: each place
  has an articulatory anchor, and a sound's formant target is the blend of its
  places.  So *e* (kaṇṭha + tālu) is literally between *a* and *i*; *ai* has the
  same places but is *vivṛtatara*, rendered as a glide a→i.  Stops are closure +
  burst (spṛṣṭa), semivowels are brief vowel-like constrictions (īṣatspṛṣṭa),
  ūṣmans are frication (īṣadvivṛta); ghoṣa sets voicing, mahāprāṇa aspiration.

The output is bit-for-bit reproducible (seeded noise).  It is a formant
synthesiser — intelligible and principled, not natural.  A neural vocoder
conditioned on this plan is the obvious next step (docs/RESEARCH.md §5.6).
Requires numpy.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field

import numpy as np

from .. import translit, varna
from ..varna import Prayatna, Sthana

SR = 16000
MATRA = 0.10                     # seconds per mātrā
F0 = 118.0

# articulatory anchors per place: (F1, F2, F3)
ANCHOR = {
    Sthana.KANTHA: (760.0, 1250.0, 2600.0),
    Sthana.TALU: (290.0, 2300.0, 3000.0),
    Sthana.OSTHA: (320.0, 760.0, 2400.0),
    Sthana.MURDHAN: (460.0, 1380.0, 1750.0),      # retroflex: low F3
    Sthana.DANTA: (420.0, 1500.0, 2700.0),
    Sthana.JIHVAMULA: (600.0, 1500.0, 2500.0),
}
# burst / frication centre per place
NOISE = {
    Sthana.KANTHA: 1900.0, Sthana.TALU: 3100.0, Sthana.MURDHAN: 2300.0,
    Sthana.DANTA: 5200.0, Sthana.OSTHA: 1000.0, Sthana.JIHVAMULA: 1500.0,
}
ACCENT = {"/": "udatta", "\\": "anudatta", "^": "svarita"}


def formants_of(v: varna.Varna) -> tuple[float, float, float]:
    places = [p for p in v.sthana if p in ANCHOR]
    if not places:
        return ANCHOR[Sthana.KANTHA]
    if len(places) == 2 and Sthana.KANTHA in places:
        # e, o: a fused blend weighted towards the non-throat place
        other = next(p for p in places if p != Sthana.KANTHA)
        w = (0.35, 0.65)
        return tuple(w[0] * a + w[1] * b for a, b in zip(ANCHOR[Sthana.KANTHA], ANCHOR[other]))
    return tuple(float(np.mean([ANCHOR[p][k] for p in places])) for k in range(3))


@dataclass
class Seg:
    dur: float
    av: float = 0.0                           # voicing amplitude
    an: float = 0.0                           # noise amplitude
    fmt: tuple = (500.0, 1500.0, 2500.0)
    fmt_end: tuple | None = None              # glide target (diphthongs)
    noise_fc: float | None = None             # band-pass centre for frication; None = through formants
    f0: tuple = (1.0, 1.0)                    # start/end multipliers
    nasal: bool = False
    label: str = ""
    accent_split: float = 0.0                 # svarita: fraction held high (1.2.32)


@dataclass
class Plan:
    segs: list[Seg] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segs)


def _tokens(slp: str):
    """(phoneme, accent|None, pluta?) tuples; spaces kept as pauses."""
    out, i = [], 0
    while i < len(slp):
        ch = slp[i]
        i += 1
        if ch in " '":
            if ch == " ":
                out.append((" ", None, False))
            continue
        acc, pluta = None, False
        while i < len(slp) and slp[i] in "/\\^3":
            if slp[i] == "3":
                pluta = True
            else:
                acc = ACCENT[slp[i]]
            i += 1
        out.append((ch, acc, pluta))
    return out


def plan(slp: str) -> Plan:
    toks = _tokens(slp)
    p = Plan()
    vowels = [t for t in toks if varna.is_vowel(t[0])]
    accented = any(t[1] for t in vowels)
    vi = 0

    def next_vowel_fmt(k: int):
        for t in toks[k + 1:]:
            if varna.is_vowel(t[0]):
                return formants_of(varna.get(t[0]))
        return ANCHOR[Sthana.KANTHA]

    prev_vowel_fmt = ANCHOR[Sthana.KANTHA]
    for k, (ch, acc, pluta) in enumerate(toks):
        if ch == " ":
            p.segs.append(Seg(0.6 * MATRA, label="pause"))
            continue
        v = varna.get(ch)
        if v.vowel:
            matra = 3 if pluta else v.matra                           # 1.2.27
            if accented:
                a = acc or "anudatta"
                f0 = {"udatta": (1.22, 1.22), "anudatta": (0.88, 0.88), "svarita": (1.22, 0.86)}[a]
            else:                                                      # gentle declination
                d = 1.08 - 0.16 * vi / max(len(vowels) - 1, 1)
                f0 = (d, d - 0.02)
            vi += 1
            fmt = formants_of(v)
            glide = None
            if v.prayatna == Prayatna.VIVRTATARA:                      # ai, au: a → i / u
                other = next(s for s in v.sthana if s != Sthana.KANTHA)
                glide, fmt = ANCHOR[other], ANCHOR[Sthana.KANTHA]
            split = (0.5 / matra) if acc == "svarita" else 0.0         # 1.2.32 ardhahrasvam
            p.segs.append(Seg(matra * MATRA, av=1.0, fmt=fmt, fmt_end=glide, f0=f0,
                              label=v.iast, accent_split=split))
            prev_vowel_fmt = glide or fmt
            continue
        nxt = next_vowel_fmt(k)
        place = next((s for s in v.sthana if s in NOISE), Sthana.KANTHA)
        half = 0.5 * MATRA
        if ch == "H":                                                  # visarga: h + echo
            p.segs.append(Seg(half, an=0.35, fmt=prev_vowel_fmt, label="ḥ"))
            if k + 1 == len(toks) or toks[k + 1][0] == " ":
                p.segs.append(Seg(0.4 * MATRA, av=0.35, fmt=prev_vowel_fmt, f0=(0.85, 0.8), label="(echo)"))
            continue
        if ch == "M":                                                  # anusvāra: nasal murmur
            p.segs.append(Seg(MATRA * 0.8, av=0.55, fmt=(260.0, prev_vowel_fmt[1] * 0.8, 2500.0),
                              nasal=True, label="ṃ"))
            continue
        if v.prayatna == Prayatna.SPRSTA and v.nasal:                  # nasal stop
            loc = ANCHOR[place]
            p.segs.append(Seg(half, av=0.5, fmt=(250.0, 0.5 * (loc[1] + nxt[1]), 2400.0),
                              nasal=True, label=v.iast))
            continue
        if v.prayatna == Prayatna.SPRSTA:                              # stop: closure + burst (+ aspiration)
            p.segs.append(Seg(0.7 * half, av=0.12 if v.ghosa else 0.0, fmt=(200.0, 900.0, 2400.0),
                              label=v.iast + " closure"))
            p.segs.append(Seg(0.012, an=0.9, noise_fc=NOISE[place], label=v.iast + " burst"))
            if v.mahaprana:
                p.segs.append(Seg(1.0 * half, av=0.45 if v.ghosa else 0.0, an=0.5, fmt=nxt,
                                  label=v.iast + " aspiration"))
            continue
        if v.prayatna == Prayatna.ISAT_SPRSTA:                         # y r l v
            f = formants_of(v)
            if ch == "v":
                f = (330.0, 1000.0, 2300.0)
            p.segs.append(Seg(half, av=0.7, fmt=f, label=v.iast))
            continue
        if ch == "h":                                                  # voiced (murmured) h
            p.segs.append(Seg(half, av=0.35, an=0.35, fmt=nxt, label="h"))
            continue
        p.segs.append(Seg(1.2 * half, an=0.55, noise_fc=NOISE[place],  # ś ṣ s, ẖ ḫ
                          av=0.0, label=v.iast))
    return p


# ─── rendering ───────────────────────────────────────────────────────────────

def _res_coeffs(f, bw):
    T = 1.0 / SR
    C = -np.exp(-2 * np.pi * bw * T)
    B = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * f * T)
    return 1 - B - C, B, C


def render(pl: Plan, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_total = int(pl.duration * SR) + 1
    out = np.zeros(n_total)
    st = [[0.0, 0.0] for _ in range(4)]          # resonator states: F1 F2 F3 + noise bp
    lp = [0.0, 0.0, 0.0]                         # glottal tilt (2 poles) + radiation
    phase = 0.0
    n0 = 0
    fmt_prev = pl.segs[0].fmt if pl.segs else (500.0, 1500.0, 2500.0)
    for s in pl.segs:
        n = int(s.dur * SR)
        for i in range(n):
            t = i / max(n - 1, 1)
            # formant trajectory: 25 ms transition from the previous target, then glide
            tr = min(1.0, i / (0.025 * SR))
            tgt = s.fmt if s.fmt_end is None else tuple(a + (b - a) * t for a, b in zip(s.fmt, s.fmt_end))
            fmt = tuple(p + (q - p) * tr for p, q in zip(fmt_prev, tgt))
            if s.accent_split and t < s.accent_split:
                f0m = s.f0[0]
            elif s.accent_split:
                u = (t - s.accent_split) / (1 - s.accent_split)
                f0m = s.f0[0] + (s.f0[1] - s.f0[0]) * u
            else:
                f0m = s.f0[0] + (s.f0[1] - s.f0[0]) * t
            f0 = F0 * f0m
            phase += f0 / SR
            pulse = 0.0
            if phase >= 1.0:
                phase -= 1.0
                pulse = 1.0
            # glottal source: two one-pole low-passes (−12 dB/oct), then a
            # differentiator for lip radiation (+6 dB/oct): zero-mean, −6 dB/oct
            lp[0] = 0.97 * lp[0] + pulse
            lp[1] = 0.97 * lp[1] + lp[0]
            voiced = s.av * (lp[1] - lp[2]) * 0.5
            lp[2] = lp[1]
            noise = rng.standard_normal() * s.an
            env = min(1.0, i / 40.0, (n - i) / 40.0)                     # click-free edges
            if s.noise_fc is not None:
                a, b, c = _res_coeffs(s.noise_fc, s.noise_fc * 0.35)
                y = a * noise + b * st[3][0] + c * st[3][1]
                st[3] = [y, st[3][0]]
                x = voiced
                fric = y * 0.6
            else:
                x = voiced + noise * 0.25
                fric = 0.0
            bws = (60.0, 90.0, 150.0) if not s.nasal else (120.0, 200.0, 250.0)
            for k in range(3):
                a, b, c = _res_coeffs(fmt[k], bws[k])
                y = a * x + b * st[k][0] + c * st[k][1]
                st[k] = [y, st[k][0]]
                x = y
            if s.nasal:
                x *= 0.6
            out[n0 + i] = (x + fric) * env
        n0 += n
        if s.av > 0.3 or s.fmt_end is not None:
            fmt_prev = s.fmt_end or s.fmt
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.85).astype(np.float64)


def write_wav(path: str, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def speak(text: str, path: str | None = None, seed: int = 0) -> np.ndarray:
    """Text (IAST / SLP1 / Devanagari; SLP1 accent marks / \\ ^ allowed) → audio."""
    accents = {ch for ch in text if ch in "/\\^"}
    slp = text if accents else translit.detect_and_parse(text)
    audio = render(plan(slp), seed=seed)
    if path:
        write_wav(path, audio)
    return audio
