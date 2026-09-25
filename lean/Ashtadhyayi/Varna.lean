/-!
# Varṇa — the sound inventory and its articulatory features

Sounds are named by their SLP1 letter (one ASCII character per phoneme):
`f F x X` are ṛ ṝ ḷ ḹ, `E O` are ai au, `w W q Q R` are ṭ ṭh ḍ ḍh ṇ, `Y N` are ñ ṅ,
`S z` are ś ṣ, `M H` are anusvāra and visarga.

The feature table is the classical Śikṣā description quoted under 1.1.9
*tulyāsyaprayatnaṃ savarṇam* (see `python/panini/varna.py` for the Sanskrit).
-/

namespace Ashtadhyayi

inductive Varna where
  | a | A | i | I | u | U | f | F | x | X | e | E | o | O
  | k | K | g | G | N | c | C | j | J | Y | w | W | q | Q | R
  | t | T | d | D | n | p | P | b | B | m
  | y | r | l | v | S | z | s | h
  | M | H
  deriving DecidableEq, Repr, Inhabited

/-- Place of articulation (sthāna). -/
inductive Sthana where
  | kantha | talu | murdhan | danta | ostha | nasika
  deriving DecidableEq, Repr

/-- Internal effort (ābhyantara prayatna). `vivrtatara` ("more open") is how the
tradition keeps ai/au from being savarṇa with e/o. -/
inductive Prayatna where
  | sprsta | isatSprsta | isadVivrta | vivrta | vivrtatara
  deriving DecidableEq, Repr

namespace Varna

def all : List Varna :=
  [a, A, i, I, u, U, f, F, x, X, e, E, o, O, k, K, g, G, N, c, C, j, J, Y, w, W, q, Q, R,
   t, T, d, D, n, p, P, b, B, m, y, r, l, v, S, z, s, h, M, H]

def vowels : List Varna := [a, A, i, I, u, U, f, F, x, X, e, E, o, O]

def isVowel (sv : Varna) : Bool := vowels.contains sv

/-- Places, in a canonical order so that list equality is set equality. -/
def sthana : Varna → List Sthana
  | a | A | k | K | g | G | h | H => [.kantha]
  | N => [.kantha, .nasika]
  | i | I | c | C | j | J | y | S => [.talu]
  | Y => [.talu, .nasika]
  | f | F | w | W | q | Q | r | z => [.murdhan]
  | R => [.murdhan, .nasika]
  | x | X | t | T | d | D | l | s => [.danta]
  | n => [.danta, .nasika]
  | u | U | p | P | b | B => [.ostha]
  | m => [.ostha, .nasika]
  | e | E => [.kantha, .talu]
  | o | O => [.kantha, .ostha]
  | v => [.danta, .ostha]
  | M => [.nasika]

def prayatna : Varna → Prayatna
  | a | A | i | I | u | U | f | F | x | X | e | o => .vivrta
  | E | O => .vivrtatara
  | y | r | l | v => .isatSprsta
  | S | z | s | h | M | H => .isadVivrta
  | _ => .sprsta

/-- Long (dīrgha) counterpart of a simple vowel. -/
def long : Varna → Varna
  | a => A | i => I | u => U | f => F | x => X | other => other

/-- Short counterpart ("base") of a simple vowel. -/
def base : Varna → Varna
  | A => a | I => i | U => u | F => f | X => x | other => other

end Varna
end Ashtadhyayi
