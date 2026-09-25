import Ashtadhyayi.Shivasutra

/-!
# Why *h* is listed twice in the Śiva-sūtras

Pāṇini lists every sound once — except *h* (sūtras 5 and 14).  Is that a
flaw?  No: it is **forced**.  Take four classes the grammar actually uses and
look only at five representative sounds a, h, b, s, y:

| class | used in | restricted to {a,h,b,s,y} |
|---|---|---|
| haś  (voiced consonants)       | 6.1.114 haśi ca            | {h, y, b} |
| jhal (obstruents)              | 8.2.39 jhalāṃ jaśo 'nte    | {b, s, h} |
| śal  (sibilants + h)           | e.g. 3.1.45 śala igupadhād aniṭaḥ ksaḥ | {s, h} |
| aṇ, second reading (1.1.69)    | 1.1.69 aṇudit savarṇasya   | {a, h, y} |

If each sound occurred once and every class were an interval of the list, the
restriction to these five sounds would also be intervals (an interval meets any
subsequence in a contiguous block).  We check all 5! orders and show that
**no** order works — so any interval coding of these classes must repeat a
sound, and Pāṇini's choice of repeating *h* suffices (Petersen 2004 proves the
full optimality result for the whole system).
-/

namespace Ashtadhyayi

def atoms : List Varna := [.a, .h, .b, .s, .y]

def restrict (cls : List Varna) : List Varna := atoms.filter (· ∈ cls)

/-- The restrictions are computed from the real pratyāhāras, not typed in. -/
theorem restrictions :
    restrict haS = [.h, .b, .y] ∧ restrict jhal = [.h, .b, .s] ∧
    restrict shal = [.h, .s] ∧ restrict (pratyahara .a .R 2) = [.a, .h, .y] := by
  decide

/-- All orderings of a list (insertion construction). -/
def insertAll (new : Varna) : List Varna → List (List Varna)
  | [] => [[new]]
  | hd :: rest => (new :: hd :: rest) :: (insertAll new rest).map (hd :: ·)

def perms : List Varna → List (List Varna)
  | [] => [[]]
  | hd :: rest => (perms rest).flatMap (insertAll hd)

/-- Is `cls` a contiguous block of `order`? (at most one run of members) -/
def contiguous (order cls : List Varna) : Bool :=
  let bs := order.map cls.contains
  (bs.zip (false :: bs)).countP (fun (cur, prev) => cur && !prev) ≤ 1

def allFourIntervals (order : List Varna) : Bool :=
  contiguous order (restrict haS) && contiguous order (restrict jhal) &&
  contiguous order (restrict shal) && contiguous order (restrict (pratyahara .a .R 2))

/-- 120 = 5! orderings, all distinct, each a rearrangement of `atoms`. -/
theorem perms_complete : (perms atoms).length = 120 ∧ (perms atoms).eraseDups.length = 120 ∧
    (perms atoms).all (fun o => atoms.all (fun sv => o.count sv == 1) && o.length == 5) := by
  decide +kernel

/-- **No ordering of a, h, b, s, y with each sound once makes all four classes intervals.** -/
theorem h_must_repeat : ∀ order ∈ perms atoms, allFourIntervals order = false := by
  decide +kernel

/-- …whereas in Pāṇini's list, with h repeated, all four are intervals
(an instance of the general theorem `pratyahara_is_interval`). -/
theorem panini_succeeds :
    (∃ pre post, pre ++ haS ++ post = sounds) ∧ (∃ pre post, pre ++ jhal ++ post = sounds) ∧
    (∃ pre post, pre ++ shal ++ post = sounds) ∧
    (∃ pre post, pre ++ pratyahara .a .R 2 ++ post = sounds) :=
  ⟨pratyahara_is_interval _ _ _, pratyahara_is_interval _ _ _,
   pratyahara_is_interval _ _ _, pratyahara_is_interval _ _ _⟩

end Ashtadhyayi
