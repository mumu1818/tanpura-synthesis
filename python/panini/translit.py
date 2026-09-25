"""Deterministic transliteration: SLP1 ⇄ IAST ⇄ Devanagari.

Writing is the easy half of "deterministic Sanskrit": the script is a pure
function of the phoneme string.  SLP1 is the internal representation because it
is one ASCII character per phoneme, which keeps every rule a string operation.
"""

from __future__ import annotations

from .varna import VARNA, VOWELS

VIRAMA = "्"
AVAGRAHA_DEVA = "ऽ"

_SLP_TO_IAST = {v.slp: v.iast for v in VARNA.values()}
_IAST_TO_SLP = {v.iast: v.slp for v in VARNA.values()}
_IAST_TO_SLP.update({"ṁ": "M", "ṛ": "f", "ṝ": "F", "ḷ": "x", "ḹ": "X"})
_IAST_KEYS = sorted(_IAST_TO_SLP, key=len, reverse=True)

_DEVA_CONSONANT = {v.deva: v.slp for v in VARNA.values() if not v.vowel and v.slp not in "MHZV"}
_DEVA_VOWEL = {v.deva: v.slp for v in VARNA.values() if v.vowel}
_DEVA_SIGN = {v.sign: v.slp for v in VARNA.values() if v.vowel and v.sign}
_DEVA_MARK = {"ं": "M", "ः": "H", "ᳵ": "Z", "ᳶ": "V"}

_PASS = set(" +-'|.,;:?!\n\t3~")


def iast_to_slp1(text: str) -> str:
    """Greedy longest-match IAST → SLP1.  'ai'/'au' are always diphthongs;
    write 'a-i' or 'aï' for a hiatus."""
    text = text.lower().replace("ï", "i").replace("ü", "u")
    out, i = [], 0
    while i < len(text):
        for key in _IAST_KEYS:
            if text.startswith(key, i):
                out.append(_IAST_TO_SLP[key])
                i += len(key)
                break
        else:
            ch = text[i]
            if ch in _PASS:
                out.append("" if ch == "-" else ch)
                i += 1
            else:
                raise ValueError(f"cannot transliterate {ch!r} in {text!r}")
    return "".join(out)


def slp1_to_iast(text: str) -> str:
    return "".join(_SLP_TO_IAST.get(ch, ch) for ch in text)


def slp1_to_devanagari(text: str) -> str:
    out: list[str] = []
    prev_consonant = False
    for ch in text:
        if ch in VOWELS:
            v = VARNA[ch]
            out.append(v.sign if prev_consonant else v.deva)
            prev_consonant = False
        elif ch in "MHZV":
            if prev_consonant:
                out.append(VIRAMA)
            out.append(VARNA[ch].deva)
            prev_consonant = False
        elif ch in VARNA:
            if prev_consonant:
                out.append(VIRAMA)
            out.append(VARNA[ch].deva)
            prev_consonant = True
        else:
            if prev_consonant:
                out.append(VIRAMA)
            prev_consonant = False
            out.append(AVAGRAHA_DEVA if ch == "'" else ch)
    if prev_consonant:
        out.append(VIRAMA)
    return "".join(out)


def devanagari_to_slp1(text: str) -> str:
    out: list[str] = []
    pending_a = False
    for ch in text:
        if ch in _DEVA_CONSONANT:
            if pending_a:
                out.append("a")
            out.append(_DEVA_CONSONANT[ch])
            pending_a = True
            continue
        if ch in _DEVA_SIGN and pending_a:
            out.append(_DEVA_SIGN[ch])
            pending_a = False
            continue
        if ch == VIRAMA:
            pending_a = False
            continue
        if pending_a:
            out.append("a")
            pending_a = False
        if ch in _DEVA_VOWEL:
            out.append(_DEVA_VOWEL[ch])
        elif ch in _DEVA_MARK:
            out.append(_DEVA_MARK[ch])
        elif ch == AVAGRAHA_DEVA:
            out.append("'")
        else:
            out.append(ch)
    if pending_a:
        out.append("a")
    return "".join(out)


def detect_and_parse(text: str) -> str:
    """Accept Devanagari, IAST or SLP1 and return SLP1."""
    if any("ऀ" <= ch <= "ॿ" for ch in text):
        return devanagari_to_slp1(text)
    if any(ord(ch) > 127 for ch in text):
        return iast_to_slp1(text)
    # Plain ASCII: SLP1 if it uses SLP1-only capitals, otherwise IAST-as-ASCII
    # (identical for words like "deva", "iti", "atra").
    if any(ch.isupper() for ch in text):
        return text
    return iast_to_slp1(text)
