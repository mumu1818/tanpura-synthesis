"""Public API: derive an utterance with a full sūtra-by-sūtra trace.

    >>> from panini import derive
    >>> derive("rāma+su gacchati").iast
    'rāmo gacchati'

Input words are either finished padas ("gacchati", "atra") or nominal stems
with an ending: "rāma+su", "rāma+bhis", or "rāma+3.3" (vibhakti.vacana).
"""

from __future__ import annotations

from dataclasses import replace

from . import samjna, translit, varna
from .engine import (Derivation, Engine, EngineError, Pada, Seg, State, Step, enumerate_choices, render,
                     render_devanagari)
from .sutras import SUTRAS

# 4.1.2 svaujasamauṭchaṣṭābhyāmbhisṅebhyāmbhyasṅasibhyāmbhyasṅasosāmṅyossup
SUP = {
    (1, 1): ("su", "su~"), (1, 2): ("au", "O"), (1, 3): ("jas", "jas"),
    (2, 1): ("am", "am"), (2, 2): ("auṭ", "Ow"), (2, 3): ("śas", "Sas"),
    (3, 2): ("bhyām", "ByAm"), (3, 3): ("bhis", "Bis"),
    (4, 2): ("bhyām", "ByAm"), (4, 3): ("bhyas", "Byas"),
    (5, 2): ("bhyām", "ByAm"), (5, 3): ("bhyas", "Byas"),
}
SUP_BY_NAME = {}
for _k, (_n, _u) in SUP.items():
    SUP_BY_NAME.setdefault(_n, _k)

TEXTS = {
    "1.2.45": ("arthavad adhātur apratyayaḥ prātipadikam", "a meaningful form that is not a root or affix is a prātipadika"),
    "4.1.2": ("svaujasamauṭchaṣṭābhyāmbhisṅebhyāmbhyasṅasibhyāmbhyasṅasosāmṅyossup", "the 21 nominal endings (sup)"),
    "1.3.2": ("upadeśe 'j anunāsika it", "a nasalised vowel in instruction is an it-marker"),
    "1.3.3": ("hal antyam", "a final consonant in instruction is an it-marker"),
    "1.3.4": ("na vibhaktau tusmāḥ", "…but not t-class, s, m at the end of a case ending"),
    "1.3.7": ("cuṭū", "initial c- and ṭ-class of an affix are it-markers"),
    "1.3.8": ("laśakv ataddhite", "initial l, ś, k-class of a non-taddhita affix are it-markers"),
    "1.3.9": ("tasya lopaḥ", "it-markers are deleted"),
    "1.4.14": ("suptiṅantaṃ padam", "what ends in sup or tiṅ is a pada"),
}


def _step(sid: str, result: str, note: str = "") -> Step:
    text, gloss = TEXTS[sid]
    return Step(sid, text, gloss, result, note)


def _parse_word(word: str) -> tuple[str, tuple[int, int] | None]:
    if "+" not in word:
        return translit.detect_and_parse(word), None
    stem, ending = word.split("+", 1)
    stem_slp = translit.detect_and_parse(stem)
    if "." in ending and ending.replace(".", "").isdigit():
        key = tuple(int(x) for x in ending.split("."))
    else:
        name = translit.slp1_to_iast(translit.detect_and_parse(ending))
        if name not in SUP_BY_NAME:
            raise EngineError(f"unknown ending {ending!r}; known: {sorted(SUP_BY_NAME)}")
        key = SUP_BY_NAME[name]
    if key not in SUP:
        raise EngineError(f"ending {key} not implemented; implemented: {sorted(SUP)}")
    if not stem_slp.endswith("a"):
        raise EngineError("only masculine a-stems are implemented for nominal derivation")
    return stem_slp, key


def build(text: str, *, gender: str = "m") -> tuple[State, list[Step], list[str]]:
    """Turn the input into segments, running the morphological prelude
    (sup introduction, it-marking and it-deletion) for stem+ending words."""
    state = State([], [])
    steps: list[Step] = []
    warnings: list[str] = []
    words = text.split()
    for p, word in enumerate(words):
        stem, key = _parse_word(word)
        if key is None:
            if stem.endswith("H"):
                warnings.append(f"'{translit.slp1_to_iast(stem)}': final ḥ read as underlying s")
                stem = stem[:-1] + "s"
            state.padas.append(Pada(stem))
            for ph in stem:
                varna.get(ph)
                state.segs.append(Seg(state.new_uid(), ph, frozenset({p})))
            continue
        vib, vac = key
        name, upadesha = SUP[key]
        state.padas.append(Pada(stem, gender=gender, derived=True))
        before = render(state)
        prefix = (before + " ") if before else ""
        steps.append(_step("1.2.45", prefix + stem, f"{translit.slp1_to_iast(stem)} is a prātipadika"))
        steps.append(_step("4.1.2", prefix + f"{stem} + {upadesha}",
                           f"vibhakti {vib}, vacana {vac}: {name}"))
        its = samjna.it_samjna(upadesha, pratyaya=True, vibhakti=True)
        plain = samjna.strip_nasal_marks(upadesha)
        for i, sid in its:
            steps.append(_step(sid, prefix + f"{stem} + {upadesha}", f"'{plain[i]}' is it"))
        if vib in (1, 2) or plain[-1] in "sm":
            last = plain[-1]
            if varna.is_consonant(last) and not any(i == len(plain) - 1 for i, _ in its):
                steps.append(_step("1.3.4", prefix + f"{stem} + {upadesha}", f"final '{last}' is not it"))
        kept = "".join(ch for i, ch in enumerate(plain) if i not in {j for j, _ in its})
        tags = frozenset({"sup", f"sup:{upadesha.replace('~', '')}", f"vibhakti:{vib}", f"vacana:{vac}"})
        for ph in stem:
            state.segs.append(Seg(state.new_uid(), ph, frozenset({p}), frozenset({"anga"})))
        for ph in kept:
            state.segs.append(Seg(state.new_uid(), ph, frozenset({p}), tags))
        if its:
            steps.append(_step("1.3.9", render(state), "it-markers deleted"))
        steps.append(_step("1.4.14", render(state), f"{translit.slp1_to_iast(stem + kept)} is a pada"))
    _coverage(state, warnings)
    return state, steps, warnings


def _coverage(state: State, warnings: list[str]) -> None:
    """Flag junctions outside the implemented fragment instead of guessing."""
    words = [p.text for p in state.padas]
    for i, w in enumerate(words):
        last = w[-1] if not state.padas[i].derived else None
        nxt = words[i + 1][0] if i + 1 < len(words) else None
        if last in ("S", "z", "h", "c", "C", "j", "J"):
            warnings.append(f"pada-final {translit.slp1_to_iast(last)} is outside the implemented rules")
        if last in ("n", "N", "R") and nxt and varna.is_consonant(nxt) and not varna.get(nxt).nasal:
            warnings.append(f"pada-final {translit.slp1_to_iast(last)} before a consonant "
                            "(8.3.7 etc.) is not implemented")
        if last in ("I", "U", "e") and nxt and varna.is_vowel(nxt):
            warnings.append(f"if '{translit.slp1_to_iast(w)}' is a dual, 1.1.11 makes it pragṛhya "
                            "(no sandhi); that is not modelled for finished padas")


def _derivation(text: str, resolver: str, gender: str, choices: dict, asiddha: bool = True) -> Derivation:
    state, steps, warnings = build(text, gender=gender)
    engine = Engine(SUTRAS, resolver=resolver, asiddha=asiddha)
    engine.run(state, choices, steps)
    return Derivation(text, render(state), steps, dict(choices), warnings)


def derive_all(text: str, *, resolver: str = "classical", gender: str = "m",
               asiddha: bool = True) -> list[Derivation]:
    """Every derivation licensed by the implemented rules (one per choice vector)."""
    return enumerate_choices(lambda ch: _derivation(text, resolver, gender, ch, asiddha))


def derive(text: str, *, resolver: str = "classical", gender: str = "m", choices: dict | None = None,
           asiddha: bool = True) -> Derivation:
    """The derivation for one choice vector (default: the preferred option at
    every vā-rule)."""
    if choices is not None:
        return _derivation(text, resolver, gender, choices, asiddha)
    return derive_all(text, resolver=resolver, gender=gender, asiddha=asiddha)[0]


def forms(text: str, **kw) -> list[str]:
    """All distinct surface results, IAST."""
    seen: list[str] = []
    for d in derive_all(text, **kw):
        if d.iast not in seen:
            seen.append(d.iast)
    return seen


def sandhi(*words: str, **kw) -> list[str]:
    return forms(" ".join(words), **kw)


def decline(stem: str, **kw) -> dict[tuple[int, int], list[str]]:
    return {key: forms(f"{stem}+{key[0]}.{key[1]}", **kw) for key in SUP}


def format_trace(d: Derivation, script: str = "iast") -> str:
    conv = {"iast": translit.slp1_to_iast, "slp1": lambda s: s,
            "deva": render_devanagari}[script]
    width = max((len(s.sutra) for s in d.steps), default=6)
    lines = []
    for s in d.steps:
        note = f"   ({s.note})" if s.note else ""
        text = s.text if len(s.text) <= 34 else s.text[:33] + "…"
        lines.append(f"{s.sutra:<{width}}  {text:<34} → {conv(s.result)}{note}")
    lines.append(f"{'':<{width}}  {'result':<34} = {d.iast}   {d.devanagari}")
    for w in d.warnings:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
