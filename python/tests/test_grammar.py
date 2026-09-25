"""Gold examples from the standard textbooks (Laghu-/Siddhānta-Kaumudī).
Run:  python -m unittest discover -s tests   (from the python/ directory)"""

import unittest

from panini import derive, derive_all, forms, sandhi, samjna, shivasutra, translit
from panini.engine import render_devanagari


class Shivasutra(unittest.TestCase):
    def test_core_pratyaharas(self):
        P = lambda n: "".join(shivasutra.pratyahara(n))
        self.assertEqual(P("ac"), "aiufxeoEO")
        self.assertEqual(P("ik"), "iufx")
        self.assertEqual(P("yaR"), "yvrl")
        self.assertEqual(P("jaS"), "jbgqd")
        self.assertEqual(P("Sal"), "Szsh")
        self.assertEqual(P("hal")[0] + P("hal")[-1], "hh")          # h occurs twice

    def test_an_is_ambiguous(self):
        self.assertEqual(shivasutra.pratyahara("aR"), ("a", "i", "u"))
        self.assertEqual(len(shivasutra.pratyahara("aR", 2)), 14)
        self.assertEqual(shivasutra.pratyahara("iR")[-1], "l")      # iṇ: second Ṇ

    def test_ac_hal_partition(self):
        ac, hal = shivasutra.members("ac", savarna=False), shivasutra.members("hal")
        self.assertFalse(ac & hal)
        self.assertEqual(ac | hal, set(shivasutra.SOUNDS))

    def test_h_is_the_only_duplicate(self):
        dup = {s for s in shivasutra.SOUNDS if shivasutra.SOUNDS.count(s) > 1}
        self.assertEqual(dup, {"h"})


class Samjna(unittest.TestCase):
    def test_guna_vrddhi_by_antaratama(self):
        self.assertEqual([samjna.rapara(samjna.guna(x), x) for x in "iufx"], ["e", "o", "ar", "al"])
        self.assertEqual([samjna.rapara(samjna.vrddhi(x), x) for x in "iuf"], ["E", "O", "Ar"])

    def test_savarna(self):
        self.assertTrue(samjna.savarna("a", "A"))
        self.assertTrue(samjna.savarna("f", "x"))                    # vārttika
        self.assertFalse(samjna.savarna("e", "E"))
        self.assertFalse(samjna.savarna("i", "S"))                   # 1.1.10 nājjhalau
        self.assertTrue(samjna.savarna("k", "N"))                    # ku includes ṅ

    def test_it_markers(self):
        f = lambda u: samjna.it_samjna(u, pratyaya=True, vibhakti=True)
        self.assertEqual(f("su~"), [(1, "1.3.2")])
        self.assertEqual(f("jas"), [(0, "1.3.7")])
        self.assertEqual(f("Sas"), [(0, "1.3.8")])
        self.assertEqual(f("Ow"), [(1, "1.3.3")])
        self.assertEqual(f("am"), [])                                # 1.3.4


class VowelSandhi(unittest.TestCase):
    CASES = {
        "deva indra": "devendra", "mahā ṛṣi": "maharṣi", "gaṅgā udaka": "gaṅgodaka",
        "deva aiśvarya": "devaiśvarya", "mahā oṣadhi": "mahauṣadhi",
        "daitya ari": "daityāri", "śrī īśa": "śrīśa", "vidhu udaya": "vidhūdaya",
        "dadhi atra": "dadhy atra", "madhu ari": "madhv ari", "pitṛ ājñā": "pitr ājñā",
        "hare atra": "hare 'tra",
    }

    def test_cases(self):
        for inp, out in self.CASES.items():
            with self.subTest(inp=inp):
                self.assertEqual(forms(inp)[0], out)

    def test_options_of_8_3_19(self):
        self.assertEqual(sandhi("hare", "iha"), ["hara iha", "haray iha"])
        self.assertEqual(sandhi("viṣṇo", "iha"), ["viṣṇa iha", "viṣṇav iha"])


class VisargaAndConsonants(unittest.TestCase):
    CASES = {
        "rāma+su gacchati": ["rāmo gacchati"],
        "rāma+su atra": ["rāmo 'tra"],
        "rāma+su tatra": ["rāmas tatra"],
        "rāma+su ca": ["rāmaś ca"],
        "rāma+su ṭīkate": ["rāmaṣ ṭīkate"],
        "rāma+su āgacchati": ["rāma āgacchati", "rāmay āgacchati"],
        "rāma+su karoti": ["rāmaḥ karoti", "rāmaẖ karoti"],
        "rāma+su śete": ["rāmaḥ śete", "rāmaś śete"],
        "devās gacchanti": ["devā gacchanti"],
        "haris ramate": ["harī ramate"],
        "punar ramate": ["punā ramate"],
        "harir gacchati": ["harir gacchati"],
        "punar tatra": ["punas tatra"],
        "rāma+am gacchati": ["rāmaṃ gacchati"],
        "jagat īśa": ["jagad īśa"],
        "tat ca": ["tac ca"],
        "tat ṭīkā": ["taṭ ṭīkā"],
        "vāk maya": ["vāṅ maya", "vāg maya"],
        "rājan atra": ["rājann atra"],
        "jagat": ["jagat", "jagad"],
    }

    def test_cases(self):
        for inp, out in self.CASES.items():
            with self.subTest(inp=inp):
                self.assertEqual(forms(inp), out)


class Subanta(unittest.TestCase):
    def test_rama(self):
        want = {"su": "rāmaḥ", "au": "rāmau", "jas": "rāmāḥ", "am": "rāmam", "auṭ": "rāmau",
                "śas": "rāmān", "bhyām": "rāmābhyām", "bhis": "rāmaiḥ", "bhyas": "rāmebhyaḥ"}
        for sup, form in want.items():
            with self.subTest(sup=sup):
                self.assertEqual(forms(f"rāma+{sup}"), [form])

    def test_trace_cites_sutras(self):
        d = derive("rāma+śas")
        cited = [s.sutra for s in d.steps]
        for sid in ("4.1.2", "1.3.8", "1.3.9", "6.1.102", "6.1.103"):
            self.assertIn(sid, cited)


class MetaRules(unittest.TestCase):
    def test_rule_conflict_resolvers(self):
        """Rajpopat (2022): 'para' = the right-hand operand.  Naive 'later rule
        wins' yields the (Vedic) *rāmebhiḥ; the tradition needs a
        pūrvavipratiṣedha to get rāmaiḥ."""
        self.assertEqual(forms("rāma+bhis", resolver="sutra-order"), ["rāmebhiḥ"])
        self.assertEqual(forms("rāma+bhis", resolver="right-operand"), ["rāmaiḥ"])
        self.assertEqual(forms("rāma+bhis", resolver="classical"), ["rāmaiḥ"])

    def test_asiddha_is_necessary(self):
        """Without 8.2.1 the engine overgenerates the textbook errors."""
        self.assertIn("hareha", forms("hare iha", asiddha=False))
        self.assertIn("devātra", forms("devās atra", asiddha=False))
        self.assertEqual(forms("rāma+su gacchati", asiddha=False), ["rāmā gacchati"])
        self.assertNotIn("hareha", forms("hare iha"))
        self.assertNotIn("devātra", forms("devās atra"))

    def test_derivation_is_a_function_of_choices(self):
        for d in derive_all("hare iha"):
            again = derive("hare iha", choices=d.choices)
            self.assertEqual(again.result, d.result)
            self.assertEqual([s.sutra for s in again.steps], [s.sutra for s in d.steps])


class Script(unittest.TestCase):
    def test_roundtrip(self):
        for w in ["rāmo gacchati", "kṛṣṇa", "jñāna", "ḍhaukate", "aiśvarya", "saṃskṛtam", "duḥkha"]:
            slp = translit.iast_to_slp1(w)
            self.assertEqual(translit.slp1_to_iast(slp), w)
            deva = translit.slp1_to_devanagari(slp)
            self.assertEqual(translit.devanagari_to_slp1(deva), slp)

    def test_devanagari(self):
        self.assertEqual(translit.slp1_to_devanagari("saMskftam"), "संस्कृतम्")
        self.assertEqual(render_devanagari("hare 'tra"), "हरेऽत्र")
        self.assertEqual(render_devanagari("rAmas tatra"), "रामस्तत्र")


if __name__ == "__main__":
    unittest.main()
