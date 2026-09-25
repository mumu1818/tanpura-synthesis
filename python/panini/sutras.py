"""The sūtras implemented by this fragment, each as a small `find` function.

Every rule states its conditions with pratyāhāras (so its domain is literally
the one Pāṇini wrote) and computes its substitute with 1.1.50 sthāne
'ntaratamaḥ where the tradition does.  Coverage is deliberately small but
exact: vowel sandhi, visarga/r/m sandhi, a few consonant rules, and the
a-stem masculine endings needed for the rule-conflict experiments.
"""

from __future__ import annotations

from typing import Iterator

from . import samjna, varna
from .engine import Match, Sutra, View
from .shivasutra import members as P, pratyahara as ORD

AC, IK, EC, AK, EN, IC = P("ac"), P("ik"), P("ec"), P("ak"), P("eN"), P("ic")
HAS, AS, HAL, JAL, KHAR, SAR = P("haS"), P("aS"), P("hal"), P("Jal"), P("Kar"), P("Sar")
YAN_, JHAY, NASAL = P("yaY"), P("Jay"), P("Yam")
AVARNA = frozenset("aA")                      # a with its savarṇa (1.1.69)
STU, SCU, ZTU = frozenset("stTdDn"), frozenset("ScCjJY"), frozenset("zwWqQR")
TU = frozenset("tTdDn")


def pairs(v: View) -> Iterator[tuple[int, object, object]]:
    for k in range(len(v) - 1):
        yield k, v[k], v[k + 1]


def _gender(v: View, e) -> str | None:
    for p in e.padas:
        g = v.state.padas[p].gender
        if g:
            return g
    return None


def _anga_final(x, y) -> bool:
    return "anga" in x.tags and "sup" not in x.tags and "sup" in y.tags


# ─── 6.1: saṃhitā (vowel sandhi) ─────────────────────────────────────────────

def f_6_1_77(v, r):
    for k, x, y in pairs(v):
        if x.ph in IK and y.ph in AC:
            yield Match(r, x.idx, x.idx, samjna.antaratama(ORD("yaR"), [x.ph]),
                        operand=x.uid, note=f"{x.ph} is ik before the ac {y.ph}")


AYADI = {"e": "ay", "o": "av", "E": "Ay", "O": "Av"}      # 1.3.10 yathāsaṅkhyam


def f_6_1_78(v, r):
    for k, x, y in pairs(v):
        if x.ph in EC and y.ph in AC:
            yield Match(r, x.idx, x.idx, AYADI[x.ph], operand=x.uid,
                        note="ec → ay/av/āy/āv respectively (1.3.10)")


def f_6_1_87(v, r):
    for k, x, y in pairs(v):
        if x.ph in AVARNA and y.ph in AC:                   # aci: anuvṛtti from 6.1.77
            g = samjna.antaratama(samjna.GUNA, [x.ph, y.ph])
            rap = frozenset({"rapara"}) if y.ph in "fFx" else frozenset()
            yield Match(r, x.idx, y.idx, samjna.rapara(g, y.ph), rap, merge=True, operand=x.uid,
                        note=f"one guṇa for both (6.1.84); nearest by 1.1.50 is {g}"
                             + (" + r by 1.1.51" if y.ph in "fF" else ""))


def f_6_1_88(v, r):
    for k, x, y in pairs(v):
        if x.ph in AVARNA and y.ph in EC:
            g = samjna.antaratama(samjna.VRDDHI, [x.ph, y.ph])
            yield Match(r, x.idx, y.idx, g, merge=True, operand=x.uid,
                        note=f"one vṛddhi for both; nearest by 1.1.50 is {g}")


def f_6_1_101(v, r):
    for k, x, y in pairs(v):
        if x.ph in AK and y.ph in AC and samjna.savarna(x.ph, y.ph):
            yield Match(r, x.idx, y.idx, samjna.dirgha(x.ph), merge=True, operand=x.uid,
                        note=f"{x.ph} and {y.ph} are savarṇa (1.1.9)")


def f_6_1_102(v, r):
    for k, x, y in pairs(v):
        if (x.ph in AK and y.ph in AC and _anga_final(x, y)
                and ("vibhakti:1" in y.tags or "vibhakti:2" in y.tags)):
            if x.ph in AVARNA and y.ph in IC:
                continue                                    # 6.1.104 nād ici
            yield Match(r, x.idx, y.idx, samjna.dirgha(x.ph), frozenset({"purvasavarna"}),
                        merge=True, operand=x.uid, note="long savarṇa of the preceding vowel")


def f_6_1_103(v, r):
    for k in range(1, len(v)):
        e, p = v[k], v[k - 1]
        if e.ph == "s" and "sup:Sas" in e.tags and "purvasavarna" in p.tags and _gender(v, e) == "m":
            yield Match(r, e.idx, e.idx, "n", operand=e.uid, note="masculine: s of śas → n")


def f_6_1_107(v, r):
    for k, x, y in pairs(v):
        if x.ph in AK and _anga_final(x, y) and "sup:am" in y.tags and y.ph == "a":
            yield Match(r, x.idx, y.idx, x.ph, merge=True, operand=x.uid,
                        note="the preceding vowel stands for both")


def f_6_1_109(v, r):
    for k, x, y in pairs(v):
        if x.ph in EN and y.ph == "a" and v.pada_final(k) and v.pada_initial(k + 1):
            yield Match(r, x.idx, y.idx, x.ph, frozenset({"avagraha"}), merge=True, operand=x.uid,
                        note="pada-final e/o absorbs a following short a (1.1.70 at = short a)")


def f_6_1_113(v, r):
    for k in range(1, len(v) - 1):
        if v[k - 1].ph == "a" and "ru" in v[k].tags and v[k + 1].ph == "a":
            yield Match(r, v[k].idx, v[k].idx, "u", operand=v[k].uid,
                        note="sees ru although 8.2.66 is asiddha: it names ru explicitly")


def f_6_1_114(v, r):
    for k in range(1, len(v) - 1):
        if v[k - 1].ph == "a" and "ru" in v[k].tags and v[k + 1].ph in HAS:
            yield Match(r, v[k].idx, v[k].idx, "u", operand=v[k].uid,
                        note=f"ru after a, before the haś {v[k + 1].ph}")


# ─── 6.3 / 7.x: aṅga operations ──────────────────────────────────────────────

def f_6_3_111(v, r):
    segs = v.state.segs
    for e in v.entries:
        nxt = segs[e.idx + 1] if e.idx + 1 < len(segs) else None
        if (e.ph in "aiu" and nxt is not None and nxt.ph == "" and nxt.hist
                and nxt.hist[-1][0] == "8.3.14"):
            yield Match(r, e.idx, e.idx, samjna.dirgha(e.ph), operand=e.uid,
                        note="r was deleted by 8.3.14; 6.3.111 names that deletion, so it sees it")


def f_7_1_9(v, r):
    for k, x, y in pairs(v):
        if x.ph == "a" and _anga_final(x, y) and "sup:Bis" in y.tags:
            j = k + 1
            while v[j + 1] is not None and "sup:Bis" in v[j + 1].tags:
                j += 1
            yield Match(r, y.idx, v[j].idx, "Es", y.tags, operand=y.uid,
                        note="whole of bhis replaced (1.1.55 anekāl śit sarvasya)")


def f_7_3_102(v, r):
    for k, x, y in pairs(v):
        if x.ph == "a" and _anga_final(x, y) and y.ph in YAN_:
            yield Match(r, x.idx, x.idx, "A", operand=x.uid, note=f"sup begins with the yañ {y.ph}")


def f_7_3_103(v, r):
    for k, x, y in pairs(v):
        if x.ph == "a" and _anga_final(x, y) and y.ph in JAL and "vacana:3" in y.tags:
            yield Match(r, x.idx, x.idx, "e", operand=x.uid,
                        note=f"plural sup beginning with the jhal {y.ph}")


# ─── tripādī ─────────────────────────────────────────────────────────────────

def f_8_2_39(v, r):
    for k in range(len(v)):
        x = v[k]
        if x.ph in JHAY and v.pada_final(k):
            yield Match(r, x.idx, x.idx, samjna.antaratama(ORD("jaS"), [x.ph]), operand=x.uid)


def f_8_2_66(v, r):
    for k in range(len(v)):
        x = v[k]
        if x.ph == "s" and v.pada_final(k):
            yield Match(r, x.idx, x.idx, "r", frozenset({"ru"}), operand=x.uid,
                        note="pada-final s → ru (u is it, 1.3.2)")


def f_8_3_14(v, r):
    for k, x, y in pairs(v):
        if x.ph == "r" and y.ph == "r":
            yield Match(r, x.idx, x.idx, "", operand=x.uid)


def f_8_3_15(v, r):
    for k in range(len(v)):
        x, y = v[k], v[k + 1]
        if x.ph == "r" and v.pada_final(k) and (y is None or y.ph in KHAR):
            yield Match(r, x.idx, x.idx, "H", operand=x.uid,
                        note="at a pause" if y is None else f"before the khar {y.ph}")


def f_8_3_17(v, r):
    for k in range(1, len(v) - 1):
        x = v[k]
        if "ru" in x.tags and v[k - 1].ph in AVARNA and v[k + 1].ph in AS:
            yield Match(r, x.idx, x.idx, "y", operand=x.uid)


def f_8_3_19(v, r):
    for k in range(1, len(v) - 1):
        x = v[k]
        if x.ph in "yv" and v.pada_final(k) and v[k - 1].ph in AVARNA and v[k + 1].ph in AS:
            yield Match(r, x.idx, x.idx, "", operand=x.uid, note="optional (Śākalya's view)")


def f_8_3_22(v, r):
    for k in range(1, len(v) - 1):
        x = v[k]
        if x.ph == "y" and v.pada_final(k) and v[k - 1].ph in AVARNA and v[k + 1].ph in HAL:
            yield Match(r, x.idx, x.idx, "", operand=x.uid)


def f_8_3_23(v, r):
    for k, x, y in pairs(v):
        if x.ph == "m" and v.pada_final(k) and y.ph in HAL:
            yield Match(r, x.idx, x.idx, "M", operand=x.uid)


def f_8_3_32(v, r):
    for k in range(1, len(v) - 1):
        x, p, y = v[k], v[k - 1], v[k + 1]
        if (x.ph in "NRn" and v.pada_final(k) and varna.is_vowel(p.ph)
                and varna.get(p.ph).matra == 1 and y.ph in AC):
            yield Match(r, y.idx, y.idx, x.ph, insert=True, operand=x.uid, padas=x.padas,
                        note="ṅamuṭ augment on the following vowel (written with the first word)")


def f_8_3_34(v, r):
    for k, x, y in pairs(v):
        if x.ph == "H" and y.ph in KHAR:
            yield Match(r, x.idx, x.idx, "s", operand=x.uid)


def f_8_3_36(v, r):
    for k, x, y in pairs(v):
        if x.ph == "H" and y.ph in SAR:
            yield Match(r, x.idx, x.idx, "H", operand=x.uid, note="visarga retained (optional)")


def f_8_3_37(v, r):
    for k, x, y in pairs(v):
        if x.ph == "H" and y.ph in "kKpP":
            alt = ["H", "Z"] if y.ph in "kK" else ["H", "V"]
            yield Match(r, x.idx, x.idx, alt, operand=x.uid,
                        note="jihvāmūlīya / upadhmānīya, or visarga by 'ca'")


SCUTVA = dict(zip("stTdDn", "ScCjJY"))
SHTUTVA = dict(zip("stTdDn", "zwWqQR"))


def f_8_4_40(v, r):
    for k in range(len(v)):
        x, p, n = v[k], v[k - 1] if k else None, v[k + 1]
        if x.ph not in STU:
            continue
        if n is not None and n.ph in SCU or p is not None and p.ph in SCU:
            if p is not None and p.ph == "S" and x.ph in TU:
                continue                                    # 8.4.44 śāt
            yield Match(r, x.idx, x.idx, SCUTVA[x.ph], operand=x.uid)


def f_8_4_41(v, r):
    for k in range(len(v)):
        x, p, n = v[k], v[k - 1] if k else None, v[k + 1]
        if x.ph not in STU:
            continue
        after = n is not None and n.ph in ZTU and not (x.ph in TU and n.ph == "z")   # 8.4.43 toḥ ṣi
        before = p is not None and p.ph in ZTU and not (p.ph != "z" and v.pada_final(k - 1))  # 8.4.42
        if after or before:
            yield Match(r, x.idx, x.idx, SHTUTVA[x.ph], operand=x.uid)


def f_8_4_45(v, r):
    for k, x, y in pairs(v):
        if x.ph in JHAY and v.pada_final(k) and y.ph in NASAL:
            yield Match(r, x.idx, x.idx, samjna.antaratama(ORD("Yam"), [x.ph]), operand=x.uid,
                        note="optional nasal before a nasal")


def f_8_4_55(v, r):
    for k, x, y in pairs(v):
        if x.ph in JAL and y.ph in KHAR:
            yield Match(r, x.idx, x.idx, samjna.antaratama(ORD("car"), [x.ph]), operand=x.uid)


def f_8_4_56(v, r):
    if len(v) and v[len(v) - 1].ph in JAL:
        x = v[len(v) - 1]
        yield Match(r, x.idx, x.idx, samjna.antaratama(ORD("car"), [x.ph]), operand=x.uid,
                    note="optional at a pause")


S = Sutra
SUTRAS: list[Sutra] = [
    S("6.1.77", "iko yaṇ aci", "i u ṛ ḷ become y v r l before a vowel", f_6_1_77),
    S("6.1.78", "eco 'yavāyāvaḥ", "e o ai au become ay av āy āv before a vowel", f_6_1_78),
    S("6.1.87", "ād guṇaḥ", "a/ā + a vowel → one guṇa vowel", f_6_1_87),
    S("6.1.88", "vṛddhir eci", "a/ā + e o ai au → one vṛddhi vowel", f_6_1_88,
      apavada_of=frozenset({"6.1.87"})),
    S("6.1.101", "akaḥ savarṇe dīrghaḥ", "a simple vowel + a savarṇa vowel → one long vowel",
      f_6_1_101, apavada_of=frozenset({"6.1.77", "6.1.87"})),
    S("6.1.102", "prathamayoḥ pūrvasavarṇaḥ",
      "before a vowel of the 1st/2nd case endings: the long form of the preceding vowel "
      "(not after a before ic, 6.1.104 nād ici)", f_6_1_102, apavada_of=frozenset({"6.1.101"})),
    S("6.1.103", "tasmāc chaso naḥ puṃsi", "after that, s of śas → n in the masculine", f_6_1_103),
    S("6.1.107", "ami pūrvaḥ", "before am, the preceding vowel alone", f_6_1_107,
      apavada_of=frozenset({"6.1.101", "6.1.102"})),
    S("6.1.109", "eṅaḥ padāntād ati", "pada-final e/o + short a → e/o (avagraha)", f_6_1_109,
      apavada_of=frozenset({"6.1.78"})),
    S("6.1.113", "ato ror aplutād aplute", "ru between short a and short a → u", f_6_1_113,
      sees=frozenset({"8.2.66"})),
    S("6.1.114", "haśi ca", "ru after short a and before a voiced consonant → u", f_6_1_114,
      sees=frozenset({"8.2.66"})),
    S("6.3.111", "ḍhralope pūrvasya dīrgho 'ṇaḥ", "a i u lengthen when a following r (ḍh) is deleted",
      f_6_3_111, sees=frozenset({"8.2.66", "8.3.14"})),
    S("7.1.9", "ato bhisa ais", "after an a-stem, bhis → ais", f_7_1_9),
    S("7.3.102", "supi ca", "a-stem final a → ā before a sup beginning with yañ", f_7_3_102),
    S("7.3.103", "bahuvacane jhaly et", "a-stem final a → e before a plural sup beginning with jhal",
      f_7_3_103, apavada_of=frozenset({"7.3.102"})),
    S("8.2.39", "jhalāṃ jaśo 'nte", "a pada-final obstruent → voiced unaspirated stop", f_8_2_39),
    S("8.2.66", "sasajuṣo ruḥ", "pada-final s → ru", f_8_2_66, apavada_of=frozenset({"8.2.39"})),
    S("8.3.14", "ro ri", "r before r is deleted", f_8_3_14),
    S("8.3.15", "kharavasānayor visarjanīyaḥ", "pada-final r → visarga before khar or a pause", f_8_3_15),
    S("8.3.17", "bhobhagoaghoapūrvasya yo 'śi", "ru after a/ā (bho, bhago, agho) → y before aś", f_8_3_17),
    S("8.3.19", "lopaḥ śākalyasya", "pada-final y/v after a/ā before aś is optionally dropped",
      f_8_3_19, optional=True),
    S("8.3.22", "hali sarveṣām", "that y is always dropped before a consonant", f_8_3_22,
      apavada_of=frozenset({"8.3.19"})),
    S("8.3.23", "mo 'nusvāraḥ", "pada-final m → anusvāra before a consonant", f_8_3_23),
    S("8.3.32", "ṅamo hrasvād aci ṅamuṇ nityam", "ṅ ṇ n after a short vowel double before a vowel", f_8_3_32),
    S("8.3.34", "visarjanīyasya saḥ", "visarga → s before khar", f_8_3_34),
    S("8.3.36", "vā śari", "before ś ṣ s the visarga may stay", f_8_3_36, optional=True, retains=True,
      apavada_of=frozenset({"8.3.34"})),
    S("8.3.37", "kupvoḥ ẖka ḫpau ca", "before k/kh, p/ph: jihvāmūlīya / upadhmānīya (or visarga)",
      f_8_3_37, apavada_of=frozenset({"8.3.34"})),
    S("8.4.40", "stoḥ ścunā ścuḥ", "s/t-class in contact with ś/c-class → ś/c-class", f_8_4_40),
    S("8.4.41", "ṣṭunā ṣṭuḥ", "s/t-class in contact with ṣ/ṭ-class → ṣ/ṭ-class", f_8_4_41),
    S("8.4.45", "yaro 'nunāsike 'nunāsiko vā", "a stop before a nasal may become the nasal of its class",
      f_8_4_45, optional=True),
    S("8.4.55", "khari ca", "an obstruent before khar → voiceless unaspirated", f_8_4_55),
    S("8.4.56", "vā 'vasāne", "the same, optionally, at a pause", f_8_4_56, optional=True),
]

BY_ID = {s.id: s for s in SUTRAS}
