"""Technical terms (saṃjñā) and the substitution metarule sthāne 'ntaratamaḥ.

Nothing in this module is a lookup table of sandhi results.  Guṇa of i is *e*
because 1.1.50 sthāne 'ntaratamaḥ picks, from the guṇa set {a, e, o}, the
member whose places of articulation are closest to those of i; the single
substitute for a + i is *e* because e's places (throat + palate) are exactly
the union of a's and i's.  The tables fall out of the feature geometry.
"""

from __future__ import annotations

from typing import Iterable

from . import varna
from .varna import Prayatna, Sthana

# 1.1.1 vṛddhir ādaic — ā, ai, au are called vṛddhi (āt is tapara: long ā only, 1.1.70)
VRDDHI = ("A", "E", "O")
# 1.1.2 adeṅ guṇaḥ — a, e, o are called guṇa (at is tapara: short a only)
GUNA = ("a", "e", "o")


def savarna(x: str, y: str) -> bool:
    """1.1.9 tulyāsyaprayatnaṃ savarṇam: same place (within the mouth) and same
    internal effort.  1.1.10 nājjhalau: a vowel and a consonant are never
    savarṇa.  Vārttika ṛḷvarṇayor mithaḥ sāvarṇyaṃ vācyam: ṛ and ḷ are savarṇa."""
    vx, vy = varna.get(x), varna.get(y)
    if vx.vowel != vy.vowel:                                       # 1.1.10
        return False
    if {varna.base_vowel(x), varna.base_vowel(y)} <= {"f", "x"}:    # vārttika
        return True
    oral = lambda v: v.sthana - {Sthana.NASIKA}                     # āsya = inside the mouth
    return oral(vx) == oral(vy) and vx.prayatna == vy.prayatna


def _feature_distance(cand: varna.Varna, target: varna.Varna) -> int:
    return sum((cand.ghosa != target.ghosa, cand.mahaprana != target.mahaprana,
                cand.nasal != target.nasal, cand.prayatna != target.prayatna))


def antaratama(candidates: Iterable[str], originals: Iterable[str]) -> str:
    """1.1.50 sthāne 'ntaratamaḥ — choose the substitute nearest to the
    original(s).  For an ekādeśa (one substitute for two sounds, 6.1.84) the
    target is the union of both originals' places.

    Nearness, lexicographically: (1) most shared places, (2) fewest places the
    originals lack, (3) closest in the remaining features (voice, aspiration,
    nasality, internal effort).  With this ordering ṛ's guṇa is *a* (to which
    1.1.51 ur aṇ raparaḥ then adds r), as the tradition requires."""
    originals = [varna.get(o) for o in originals]
    places: frozenset[Sthana] = frozenset().union(*(o.sthana for o in originals))
    ref = originals[-1]

    def score(c: str):
        v = varna.get(c)
        return (-len(v.sthana & places), len(v.sthana - places), _feature_distance(v, ref))

    cands = list(candidates)
    return min(cands, key=lambda c: (score(c), cands.index(c)))


def guna(x: str) -> str:
    """Guṇa substitute of an ik vowel (1.1.3 iko guṇavṛddhī + 1.1.50)."""
    return antaratama(GUNA, [x])


def vrddhi(x: str) -> str:
    return antaratama(VRDDHI, [x])


def dirgha(x: str) -> str:
    """Long savarṇa of a simple vowel (dīrgha, 1.2.27)."""
    return varna.lengthen(varna.base_vowel(x))


def rapara(sub: str, original: str) -> str:
    """1.1.51 ur aṇ raparaḥ — an aṇ (a/i/u) substituted for ṛ is followed by r
    (ḷ: by l, per the commentary on the same sūtra)."""
    base = varna.base_vowel(original)
    if base == "f" and varna.base_vowel(sub) in "aiu":
        return sub + "r"
    if base == "x" and varna.base_vowel(sub) in "aiu":
        return sub + "l"
    return sub


def it_samjna(upadesha: str, *, pratyaya: bool, vibhakti: bool, taddhita: bool = False) -> list[tuple[int, str]]:
    """Return [(index, sūtra)] of the it-markers of an element in upadeśa.

    ``upadesha`` is SLP1 with '~' after a nasalised (anunāsika) vowel.
    Implements 1.3.2 upadeśe 'j anunāsika it, 1.3.3 halantyam,
    1.3.4 na vibhaktau tusmāḥ, 1.3.7 cuṭū, 1.3.8 laśakvataddhite."""
    phon: list[tuple[str, bool]] = []                 # (phoneme, nasalised?)
    for ch in upadesha:
        if ch == "~":
            phon[-1] = (phon[-1][0], True)
        else:
            phon.append((ch, False))
    its: list[tuple[int, str]] = []
    for i, (p, nasal) in enumerate(phon):
        if varna.is_vowel(p) and nasal:
            its.append((i, "1.3.2"))
    last = len(phon) - 1
    lp = phon[last][0]
    if varna.is_consonant(lp):
        tusmah = lp in "tTdDns" or lp == "m"
        if vibhakti and tusmah:
            pass                                      # 1.3.4 blocks 1.3.3
        else:
            its.append((last, "1.3.3"))
    if pratyaya and last > 0:
        first = phon[0][0]
        if first in "cCjJYwWqQR":
            its.append((0, "1.3.7"))
        elif not taddhita and (first in "lS" or first in "kKgGN"):
            its.append((0, "1.3.8"))
    return sorted(set(its))


def strip_nasal_marks(upadesha: str) -> str:
    return upadesha.replace("~", "")


def describe(x: str) -> str:
    v = varna.get(x)
    places = " + ".join(sorted(s.value for s in v.sthana))
    extra = []
    if not v.vowel:
        extra.append("ghoṣa" if v.ghosa else "aghoṣa")
        extra.append("mahāprāṇa" if v.mahaprana else "alpaprāṇa")
    else:
        extra.append({1: "hrasva", 2: "dīrgha", 3: "pluta"}[v.matra])
    return f"{v.iast}: {places}; {v.prayatna.value}; {', '.join(extra)}"


__all__ = ["VRDDHI", "GUNA", "savarna", "antaratama", "guna", "vrddhi", "dirgha",
           "rapara", "it_samjna", "strip_nasal_marks", "describe", "Prayatna"]
