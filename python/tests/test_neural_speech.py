import unittest

try:
    import numpy as np
except ImportError:                                   # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy not installed")
class Compiled(unittest.TestCase):
    def test_network_equals_engine_on_all_vowel_pairs(self):
        from panini.neural import compiled
        ok, n, bad = compiled.verify()
        self.assertEqual(bad, [])
        self.assertEqual(ok, n)

    def test_every_used_pratyahara_is_one_interval(self):
        from panini.neural import compiled
        from panini.shivasutra import SOUNDS, USED, pratyahara
        for name in USED:
            self.assertEqual(compiled.runs(SOUNDS, frozenset(pratyahara(name))), 1, name)

    def test_sivasutra_beats_random_orders(self):
        from panini.neural import compiled
        c = compiled.sivasutra_vs_random(n_random=30)
        self.assertLess(c["sivasutra_units"], c["random_min_units"])


class Env(unittest.TestCase):
    def test_licensed_action_space(self):
        from panini.neural.env import PrakriyaEnv, reachable
        self.assertEqual(set(reachable("rāma+bhis")), {"rAmEH", "rAmeBiH"})
        env = PrakriyaEnv("rāma+bhis", resolver="right-operand")
        self.assertEqual(env.legal_actions()[env.oracle()].sutra, "7.1.9")


@unittest.skipIf(np is None, "numpy not installed")
class Speech(unittest.TestCase):
    def test_deterministic_and_timed_by_1_2_27(self):
        from panini.speech import synth
        a, b = synth.speak("rāmo gacchati"), synth.speak("rāmo gacchati")
        self.assertTrue(np.array_equal(a, b))
        short = synth.plan("a").duration
        long_ = synth.plan("A").duration
        pluta = synth.plan("a3").duration
        self.assertAlmostEqual(long_, 2 * short)
        self.assertAlmostEqual(pluta, 3 * short)

    def test_e_lies_between_a_and_i(self):
        from panini import varna
        from panini.speech.synth import formants_of
        a, e, i = (formants_of(varna.get(x)) for x in "aei")
        self.assertTrue(i[0] < e[0] < a[0] and a[1] < e[1] < i[1])

    def test_svarita_starts_high(self):
        from panini.speech.synth import plan
        s = plan("a^").segs[0]
        self.assertGreater(s.f0[0], s.f0[1])
        self.assertAlmostEqual(s.accent_split, 0.5)                 # 1.2.32: half a hrasva


if __name__ == "__main__":
    unittest.main()
