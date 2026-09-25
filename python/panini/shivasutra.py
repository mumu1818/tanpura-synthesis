"""The fourteen Māheśvara / Śiva-sūtras and pratyāhāra expansion.

1.1.71 ādir antyena sahetā — "an initial sound together with a final it-marker
denotes itself and everything in between".  A pratyāhāra is therefore an
*interval* of this list, which is what makes it such an efficient code for
natural classes (see ``panini.neural.compiled`` for why that matters to a
neural network, and ``lean/Ashtadhyayi/Shivasutra.lean`` for the proofs).
"""

from __future__ import annotations

from functools import lru_cache

from . import varna

# (sounds, it-marker), SLP1
SHIVA_SUTRAS: list[tuple[str, str]] = [
    ("aiu", "R"),        # 1  a i u Ṇ
    ("fx", "k"),         # 2  ṛ ḷ K
    ("eo", "N"),         # 3  e o Ṅ
    ("EO", "c"),         # 4  ai au C
    ("hyvr", "w"),       # 5  ha ya va ra Ṭ
    ("l", "R"),          # 6  la Ṇ
    ("YmNRn", "m"),      # 7  ña ma ṅa ṇa na M
    ("JB", "Y"),         # 8  jha bha Ñ
    ("GQD", "z"),        # 9  gha ḍha dha Ṣ
    ("jbgqd", "S"),      # 10 ja ba ga ḍa da Ś
    ("KPCWTcwt", "v"),   # 11 kha pha cha ṭha tha ca ṭa ta V
    ("kp", "y"),         # 12 ka pa Y
    ("Szs", "r"),        # 13 śa ṣa sa R
    ("h", "l"),          # 14 ha L
]

# Flat sequence of ("snd", x) / ("it", x) items.
ITEMS: list[tuple[str, str]] = []
for _sounds, _marker in SHIVA_SUTRAS:
    ITEMS.extend(("snd", s) for s in _sounds)
    ITEMS.append(("it", _marker))

SOUNDS: list[str] = [x for kind, x in ITEMS if kind == "snd"]

# The only genuine ambiguity: Ṇ closes both sūtra 1 and sūtra 6.  Tradition:
# aṇ uses the first Ṇ except in 1.1.69 aṇudit savarṇasya cāpratyayaḥ; iṇ uses
# the second (e.g. 8.3.57 iṇkoḥ).  yaṇ can only reach the second.
DEFAULT_MARKER_OCCURRENCE = {("a", "R"): 1, ("i", "R"): 2}


def parse_name(name: str) -> tuple[str, str]:
    """'ac' → ('a','c'); 'hal' → ('h','l'); 'yaR' → ('y','R')."""
    if len(name) < 2:
        raise ValueError(name)
    first = name[0]
    rest = name[1:]
    if varna.is_consonant(first):
        if not rest.startswith("a"):
            raise ValueError(f"consonant-initial pratyāhāra must be written with 'a': {name}")
        rest = rest[1:]
    if len(rest) != 1:
        raise ValueError(f"bad pratyāhāra name {name!r}")
    return first, rest


@lru_cache(maxsize=None)
def pratyahara(name: str, occurrence: int | None = None) -> tuple[str, ...]:
    """Expand a pratyāhāra (SLP1 name) to the sounds it denotes, in order.

    ``occurrence`` selects which instance of the closing it-marker is meant
    (1-based); by default the traditional reading is used."""
    start, marker = parse_name(name)
    if occurrence is None:
        occurrence = DEFAULT_MARKER_OCCURRENCE.get((start, marker), 1)
    try:
        i = ITEMS.index(("snd", start))
    except ValueError:
        raise ValueError(f"{start!r} does not begin any Śiva-sūtra item") from None
    out: list[str] = []
    seen = 0
    for kind, x in ITEMS[i:]:
        if kind == "snd":
            out.append(x)
        elif x == marker:
            seen += 1
            if seen == occurrence:
                return tuple(out)
    raise ValueError(f"marker {marker!r} occurrence {occurrence} not found after {start!r}")


def members(name: str, occurrence: int | None = None, savarna: bool = True) -> frozenset[str]:
    """Set of sounds denoted, including savarṇa long vowels by
    1.1.69 aṇudit savarṇasya cāpratyayaḥ (a denotes ā, i denotes ī, ...)."""
    base = set(pratyahara(name, occurrence))
    if savarna:
        base |= {varna.lengthen(v) for v in base if v in varna.SHORT_TO_LONG}
    return frozenset(base)


def is_in(ph: str, name: str) -> bool:
    return ph in members(name)


# Pratyāhāras commonly listed as used in the Aṣṭādhyāyī (the traditional
# count is ~42; lists differ slightly between sources).
USED = [
    "aR", "ak", "ac", "aw", "am", "aS", "al", "ik", "ic", "iR", "uk", "eN",
    "ec", "Ec", "hal", "haS", "yaR", "yam", "yaY", "yay", "yar", "Yam", "Nam",
    "Jaz", "JaS", "Jay", "Jar", "Jal", "Baz", "jaS", "baS", "Cav", "Kay",
    "Kar", "cay", "car", "Sar", "Sal", "may", "ral", "vaS", "val",
]
