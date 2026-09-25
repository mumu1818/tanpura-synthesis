import Ashtadhyayi.Varna

/-!
# The Śiva-sūtras and 1.1.71 *ādir antyena sahetā*

A pratyāhāra names a start sound and a closing it-marker; it denotes the start
and every sound up to that marker.  We prove, for **every** start, marker and
marker-occurrence, that the result is a contiguous interval of the Śiva-sūtra
sound sequence — the property that makes pratyāhāras such an efficient code
for natural classes.
-/

namespace Ashtadhyayi
open Varna (all)

inductive Item where
  | snd (sv : Varna)
  | it (sv : Varna)
  deriving DecidableEq, Repr

/-- The fourteen Māheśvara-sūtras: (sounds, it-marker). -/
def shivasutras : List (List Varna × Varna) :=
  [ ([.a, .i, .u], .R), ([.f, .x], .k), ([.e, .o], .N), ([.E, .O], .c),
    ([.h, .y, .v, .r], .w), ([.l], .R), ([.Y, .m, .N, .R, .n], .m), ([.J, .B], .Y),
    ([.G, .Q, .D], .z), ([.j, .b, .g, .q, .d], .S), ([.K, .P, .C, .W, .T, .c, .w, .t], .v),
    ([.k, .p], .y), ([.S, .z, .s], .r), ([.h], .l) ]

def items : List Item :=
  shivasutras.flatMap fun (vs, mk) => vs.map Item.snd ++ [Item.it mk]

def soundsOf : List Item → List Varna
  | [] => []
  | .snd sv :: rest => sv :: soundsOf rest
  | .it _ :: rest => soundsOf rest

def sounds : List Varna := soundsOf items

/-- Skip to the first occurrence of the start sound. -/
def dropToStart : List Item → Varna → List Item
  | [], _ => []
  | .snd sv :: rest, st => if sv = st then .snd sv :: rest else dropToStart rest st
  | .it _ :: rest, st => dropToStart rest st

/-- Collect sounds until the `k`-th occurrence of the marker. -/
def takeUntilMarker : List Item → Varna → Nat → List Varna
  | [], _, _ => []
  | .snd sv :: rest, mk, k => sv :: takeUntilMarker rest mk k
  | .it iv :: rest, mk, k =>
      if iv = mk then (if k ≤ 1 then [] else takeUntilMarker rest mk (k - 1))
      else takeUntilMarker rest mk k

/-- **1.1.71 ādir antyena sahetā.** -/
def pratyahara (start marker : Varna) (occ : Nat := 1) : List Varna :=
  takeUntilMarker (dropToStart items start) marker occ

-- The classes used below
def ac := pratyahara .a .c
def hal := pratyahara .h .l
def ik := pratyahara .i .k
def ak := pratyahara .a .k
def ec := pratyahara .e .c
def yaR := pratyahara .y .R
def haS := pratyahara .h .S
def jhal := pratyahara .J .l
def shal := pratyahara .S .l
def jaS := pratyahara .j .S
def khar := pratyahara .K .r

/-! ## Computations -/

theorem ac_eq : ac = [.a, .i, .u, .f, .x, .e, .o, .E, .O] := by decide
theorem ik_eq : ik = [.i, .u, .f, .x] := by decide
theorem yaR_eq : yaR = [.y, .v, .r, .l] := by decide
theorem jaS_eq : jaS = [.j, .b, .g, .q, .d] := by decide
theorem shal_eq : shal = [.S, .z, .s, .h] := by decide

/-- `hal` begins and ends with h: the only sound Pāṇini lists twice. -/
theorem hal_ends : hal.head? = some .h ∧ hal.getLast? = some .h := by decide

/-- Vowels and consonants partition the Śiva-sūtra sounds. -/
theorem ac_hal_partition : ∀ sv ∈ sounds, (sv ∈ ac ↔ sv ∉ hal) := by decide

theorem h_only_duplicate : ∀ sv ∈ Varna.all, sounds.count sv ≤ 1 ∨ sv = .h := by decide

/-- The one real ambiguity: Ṇ closes both sūtra 1 and sūtra 6, so *aṇ* has two
readings (a i u, or a … l).  Tradition resolves it per rule (1.1.69 uses the second). -/
theorem aR_ambiguous : pratyahara .a .R 1 = [.a, .i, .u] ∧
    pratyahara .a .R 2 = [.a, .i, .u, .f, .x, .e, .o, .E, .O, .h, .y, .v, .r, .l] := by decide

/-! ## Every pratyāhāra is an interval (general theorem) -/

theorem soundsOf_append (l₁ l₂ : List Item) :
    soundsOf (l₁ ++ l₂) = soundsOf l₁ ++ soundsOf l₂ := by
  induction l₁ with
  | nil => rfl
  | cons it rest ih => cases it <;> simp [soundsOf, ih]

theorem dropToStart_suffix (L : List Item) (st : Varna) :
    ∃ pre, pre ++ dropToStart L st = L := by
  induction L with
  | nil => exact ⟨[], rfl⟩
  | cons it rest ih =>
    cases it with
    | snd sv =>
      by_cases hv : sv = st
      · exact ⟨[], by simp [dropToStart, hv]⟩
      · obtain ⟨pre, hpre⟩ := ih
        exact ⟨.snd sv :: pre, by simp [dropToStart, hv, hpre]⟩
    | it iv =>
      obtain ⟨pre, hpre⟩ := ih
      exact ⟨.it iv :: pre, by simp [dropToStart, hpre]⟩

theorem takeUntilMarker_prefix (L : List Item) (mk : Varna) (k : Nat) :
    ∃ t, takeUntilMarker L mk k ++ t = soundsOf L := by
  induction L generalizing k with
  | nil => exact ⟨[], rfl⟩
  | cons it rest ih =>
    cases it with
    | snd sv =>
      obtain ⟨t, ht⟩ := ih k
      exact ⟨t, by simp [takeUntilMarker, soundsOf, ht]⟩
    | it iv =>
      by_cases hv : iv = mk
      · by_cases hk : k ≤ 1
        · exact ⟨soundsOf rest, by simp [takeUntilMarker, soundsOf, hv, hk]⟩
        · obtain ⟨t, ht⟩ := ih (k - 1)
          exact ⟨t, by simp [takeUntilMarker, soundsOf, hv, hk, ht]⟩
      · obtain ⟨t, ht⟩ := ih k
        exact ⟨t, by simp [takeUntilMarker, soundsOf, hv, ht]⟩

/-- **1.1.71 as a theorem**: for any start sound, any marker and any occurrence,
the pratyāhāra is a contiguous block of the Śiva-sūtra sound sequence. -/
theorem pratyahara_is_interval (st mk : Varna) (k : Nat) :
    ∃ pre post, pre ++ pratyahara st mk k ++ post = sounds := by
  obtain ⟨preI, hpre⟩ := dropToStart_suffix items st
  obtain ⟨t, ht⟩ := takeUntilMarker_prefix (dropToStart items st) mk k
  refine ⟨soundsOf preI, t, ?_⟩
  rw [List.append_assoc, pratyahara, ht, ← soundsOf_append, hpre]
  rfl

end Ashtadhyayi
