import Ashtadhyayi.Samjna

/-!
# Vowel sandhi (6.1.77, 6.1.78, 6.1.87, 6.1.88, 6.1.101) as a rule system

Each sūtra is data: an applicability test phrased with pratyāhāras and a
substitute computed by 1.1.50.  Conflicts are settled only by the utsarga–
apavāda (general rule – exception) relation.  We then prove:

* **conflict-freeness** — once exceptions block their general rules, exactly
  one sūtra remains for every vowel pair, so 1.4.2 is never needed here;
* **hiatus elimination** — the output never contains two adjacent vowels;
* the textbook results, and that the exception relation is not vacuous.
-/

namespace Ashtadhyayi
open Varna

/-- Membership with savarṇa extension (1.1.69 aṇudit savarṇasya cāpratyayaḥ):
the short vowels of a pratyāhāra also denote their long forms. -/
def inP (cls : List Varna) (sv : Varna) : Bool := cls.contains sv.base || cls.contains sv

def avarna (sv : Varna) : Bool := sv.base == .a

structure Rule where
  id : String
  applies : Varna → Varna → Bool
  result : Varna → Varna → List Varna
  apavadaOf : List String

def ayadi : Varna → List Varna
  | .e => [.a, .y] | .o => [.a, .v] | .E => [.A, .y] | .O => [.A, .v] | other => [other]

def r6_1_77 : Rule := ⟨"6.1.77", fun v1 v2 => inP ik v1 && inP ac v2,
  fun v1 v2 => [antaratama yaR [v1], v2], []⟩
def r6_1_78 : Rule := ⟨"6.1.78", fun v1 v2 => inP ec v1 && inP ac v2,
  fun v1 v2 => ayadi v1 ++ [v2], []⟩
def r6_1_87 : Rule := ⟨"6.1.87", fun v1 v2 => avarna v1 && inP ac v2,
  fun v1 v2 => rapara (antaratama gunaSet [v1, v2]) v2, []⟩
def r6_1_88 : Rule := ⟨"6.1.88", fun v1 v2 => avarna v1 && inP ec v2,
  fun v1 v2 => [antaratama vrddhiSet [v1, v2]], ["6.1.87"]⟩
def r6_1_101 : Rule := ⟨"6.1.101", fun v1 v2 => inP ak v1 && inP ac v2 && savarna v1 v2,
  fun v1 _ => [v1.base.long], ["6.1.77", "6.1.87"]⟩

def acRules : List Rule := [r6_1_77, r6_1_78, r6_1_87, r6_1_88, r6_1_101]

def applicable (v1 v2 : Varna) : List Rule := acRules.filter (·.applies v1 v2)

/-- Utsarga–apavāda: a rule is blocked when an applicable rule is its exception. -/
def winners (v1 v2 : Varna) : List Rule :=
  let app := applicable v1 v2
  app.filter fun rule => !(app.any fun exc => exc.apavadaOf.contains rule.id)

def acSandhi (v1 v2 : Varna) : List Varna :=
  match winners v1 v2 with
  | [rule] => rule.result v1 v2
  | _ => [v1, v2]

/-- **Conflict-freeness**: for every pair of vowels exactly one sūtra survives. -/
theorem conflict_free : ∀ v1 ∈ Varna.vowels, ∀ v2 ∈ Varna.vowels, (winners v1 v2).length = 1 := by
  decide

def noHiatus : List Varna → Bool
  | v1 :: v2 :: rest => !(v1.isVowel && v2.isVowel) && noHiatus (v2 :: rest)
  | _ => true

/-- **Hiatus elimination**: vowel sandhi never leaves two vowels in contact. -/
theorem hiatus_free : ∀ v1 ∈ Varna.vowels, ∀ v2 ∈ Varna.vowels, noHiatus (acSandhi v1 v2) = true := by
  decide

/-- The exceptions are not vacuous: both rules really apply, and the exception wins. -/
theorem apavada_needed :
    r6_1_77.applies .i .i = true ∧ r6_1_101.applies .i .i = true ∧ acSandhi .i .i = [.I] ∧
    r6_1_87.applies .a .e = true ∧ r6_1_88.applies .a .e = true ∧ acSandhi .a .e = [.E] := by
  decide

theorem textbook :
    acSandhi .a .i = [.e] ∧          -- deva + indra → devendra
    acSandhi .A .f = [.a, .r] ∧      -- mahā + ṛṣi → maharṣi
    acSandhi .a .O = [.O] ∧          -- mahā + auṣadha → mahauṣadha
    acSandhi .a .a = [.A] ∧          -- daitya + ari → daityāri
    acSandhi .i .a = [.y, .a] ∧      -- dadhi + atra → dadhy atra
    acSandhi .u .a = [.v, .a] ∧      -- madhu + ari → madhv ari
    acSandhi .e .a = [.a, .y, .a] ∧  -- ne + ana → nayana
    acSandhi .O .a = [.A, .v, .a] := -- pau + aka → pāvaka
  by decide

end Ashtadhyayi
