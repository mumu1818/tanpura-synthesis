"""Varṇa (phoneme) inventory and Pāṇinian articulatory features.

Every sound is written internally in SLP1, the one-character-per-phoneme ASCII
scheme used by most Sanskrit software (Vidyut, the Cologne lexicon, ...).

The feature table below is the classical Śikṣā description as summarised in the
Siddhānta-Kaumudī under 1.1.9 *tulyāsyaprayatnaṃ savarṇam*:

    akuhavisarjanīyānāṃ kaṇṭhaḥ          a, ku, h, visarga      → throat
    icuyaśānāṃ tālu                      i, cu, y, ś            → palate
    ṛṭuraṣāṇāṃ mūrdhā                     ṛ, ṭu, r, ṣ            → cerebral
    ḷtulasānāṃ dantāḥ                    ḷ, tu, l, s            → teeth
    upūpadhmānīyānām oṣṭhau              u, pu, upadhmānīya     → lips
    ñamaṅaṇanānāṃ nāsikā ca              ñ m ṅ ṇ n              → + nose
    edaitoḥ kaṇṭhatālu                   e, ai                  → throat+palate
    odautoḥ kaṇṭhoṣṭham                  o, au                  → throat+lips
    vakārasya dantoṣṭham                 v                      → teeth+lips
    jihvāmūlīyasya jihvāmūlam            jihvāmūlīya            → tongue root
    nāsikā 'nusvārasya                   anusvāra               → nose

The *same* feature table drives three otherwise unrelated things in this
project: the savarṇa relation (1.1.9), the choice of the "nearest" substitute
(1.1.50 sthāne 'ntaratamaḥ), and the acoustic targets of the speech synthesiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Sthana(str, Enum):
    """Place of articulation (sthāna)."""

    KANTHA = "kaṇṭha"          # throat
    TALU = "tālu"              # palate
    MURDHAN = "mūrdhan"        # roof / cerebral
    DANTA = "danta"            # teeth
    OSTHA = "oṣṭha"            # lips
    NASIKA = "nāsikā"          # nose
    JIHVAMULA = "jihvāmūla"    # root of the tongue


class Prayatna(str, Enum):
    """Internal effort (ābhyantara prayatna)."""

    SPRSTA = "spṛṣṭa"                # full contact: stops and nasals
    ISAT_SPRSTA = "īṣatspṛṣṭa"       # slight contact: y r l v
    ISAD_VIVRTA = "īṣadvivṛta"       # slightly open: ś ṣ s h
    VIVRTA = "vivṛta"                # open: vowels
    VIVRTATARA = "vivṛtatara"        # more open: ai, au (keeps them non-savarṇa with e, o)


K, T, M, D, O, N, J = (Sthana.KANTHA, Sthana.TALU, Sthana.MURDHAN, Sthana.DANTA,
                       Sthana.OSTHA, Sthana.NASIKA, Sthana.JIHVAMULA)


@dataclass(frozen=True)
class Varna:
    slp: str                       # SLP1 character
    iast: str
    deva: str                      # independent Devanagari letter
    sthana: frozenset[Sthana]
    prayatna: Prayatna
    vowel: bool = False
    matra: int = 0                 # 1 hrasva, 2 dīrgha (1.2.27); 0 for consonants
    ghosa: bool = False            # voiced (nāda / saṃvāra)
    mahaprana: bool = False        # aspirated
    nasal: bool = False
    sign: str = ""                 # dependent vowel sign (mātrā) for Devanagari

    @property
    def is_consonant(self) -> bool:
        return not self.vowel


def _v(slp, iast, deva, sthana, matra, sign, prayatna=Prayatna.VIVRTA):
    return Varna(slp, iast, deva, frozenset(sthana), prayatna, vowel=True,
                 matra=matra, ghosa=True, sign=sign)


def _c(slp, iast, deva, sthana, prayatna, ghosa=False, mahaprana=False, nasal=False):
    return Varna(slp, iast, deva, frozenset(sthana), prayatna, ghosa=ghosa,
                 mahaprana=mahaprana, nasal=nasal)


S, IS, ID = Prayatna.SPRSTA, Prayatna.ISAT_SPRSTA, Prayatna.ISAD_VIVRTA

_TABLE: list[Varna] = [
    # vowels (ac)
    _v("a", "a", "अ", {K}, 1, ""),
    _v("A", "ā", "आ", {K}, 2, "ा"),
    _v("i", "i", "इ", {T}, 1, "ि"),
    _v("I", "ī", "ई", {T}, 2, "ी"),
    _v("u", "u", "उ", {O}, 1, "ु"),
    _v("U", "ū", "ऊ", {O}, 2, "ू"),
    _v("f", "ṛ", "ऋ", {M}, 1, "ृ"),
    _v("F", "ṝ", "ॠ", {M}, 2, "ॄ"),
    _v("x", "ḷ", "ऌ", {D}, 1, "ॢ"),
    _v("X", "ḹ", "ॡ", {D}, 2, "ॣ"),
    _v("e", "e", "ए", {K, T}, 2, "े"),
    _v("E", "ai", "ऐ", {K, T}, 2, "ै", Prayatna.VIVRTATARA),
    _v("o", "o", "ओ", {K, O}, 2, "ो"),
    _v("O", "au", "औ", {K, O}, 2, "ौ", Prayatna.VIVRTATARA),
    # stops: ku cu ṭu tu pu
    _c("k", "k", "क", {K}, S), _c("K", "kh", "ख", {K}, S, mahaprana=True),
    _c("g", "g", "ग", {K}, S, ghosa=True), _c("G", "gh", "घ", {K}, S, ghosa=True, mahaprana=True),
    _c("N", "ṅ", "ङ", {K, N}, S, ghosa=True, nasal=True),
    _c("c", "c", "च", {T}, S), _c("C", "ch", "छ", {T}, S, mahaprana=True),
    _c("j", "j", "ज", {T}, S, ghosa=True), _c("J", "jh", "झ", {T}, S, ghosa=True, mahaprana=True),
    _c("Y", "ñ", "ञ", {T, N}, S, ghosa=True, nasal=True),
    _c("w", "ṭ", "ट", {M}, S), _c("W", "ṭh", "ठ", {M}, S, mahaprana=True),
    _c("q", "ḍ", "ड", {M}, S, ghosa=True), _c("Q", "ḍh", "ढ", {M}, S, ghosa=True, mahaprana=True),
    _c("R", "ṇ", "ण", {M, N}, S, ghosa=True, nasal=True),
    _c("t", "t", "त", {D}, S), _c("T", "th", "थ", {D}, S, mahaprana=True),
    _c("d", "d", "द", {D}, S, ghosa=True), _c("D", "dh", "ध", {D}, S, ghosa=True, mahaprana=True),
    _c("n", "n", "न", {D, N}, S, ghosa=True, nasal=True),
    _c("p", "p", "प", {O}, S), _c("P", "ph", "फ", {O}, S, mahaprana=True),
    _c("b", "b", "ब", {O}, S, ghosa=True), _c("B", "bh", "भ", {O}, S, ghosa=True, mahaprana=True),
    _c("m", "m", "म", {O, N}, S, ghosa=True, nasal=True),
    # antaḥstha (semivowels)
    _c("y", "y", "य", {T}, IS, ghosa=True),
    _c("r", "r", "र", {M}, IS, ghosa=True),
    _c("l", "l", "ल", {D}, IS, ghosa=True),
    _c("v", "v", "व", {D, O}, IS, ghosa=True),
    # ūṣman (sibilants and h)
    _c("S", "ś", "श", {T}, ID, mahaprana=True),
    _c("z", "ṣ", "ष", {M}, ID, mahaprana=True),
    _c("s", "s", "स", {D}, ID, mahaprana=True),
    _c("h", "h", "ह", {K}, ID, ghosa=True, mahaprana=True),
    # ayogavāha
    _c("M", "ṃ", "ं", {N}, ID, ghosa=True, nasal=True),
    _c("H", "ḥ", "ः", {K}, ID, mahaprana=True),
    _c("Z", "ẖ", "ᳵ", {J}, ID, mahaprana=True),
    _c("V", "ḫ", "ᳶ", {O}, ID, mahaprana=True),
]

VARNA: dict[str, Varna] = {v.slp: v for v in _TABLE}

VOWELS = "aAiIuUfFxXeEoO"
SHORT_TO_LONG = {"a": "A", "i": "I", "u": "U", "f": "F", "x": "X"}
LONG_TO_SHORT = {v: k for k, v in SHORT_TO_LONG.items()}
AYOGAVAHA = "MHZV"


def get(ph: str) -> Varna:
    try:
        return VARNA[ph]
    except KeyError:
        raise ValueError(f"unknown SLP1 phoneme {ph!r}") from None


def is_vowel(ph: str) -> bool:
    return ph in VOWELS and ph != ""


def is_consonant(ph: str) -> bool:
    return ph in VARNA and not VARNA[ph].vowel


def base_vowel(ph: str) -> str:
    """Map a long simple vowel to its short counterpart (ā → a)."""
    return LONG_TO_SHORT.get(ph, ph)


def lengthen(ph: str) -> str:
    return SHORT_TO_LONG.get(ph, ph)
