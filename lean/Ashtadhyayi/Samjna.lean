import Ashtadhyayi.Shivasutra

/-!
# Saṃjñā: savarṇa, guṇa, vṛddhi and 1.1.50 *sthāne 'ntaratamaḥ*

Guṇa of *i* is *e*, and the single substitute of *a + i* is *e*, not because a
table says so but because 1.1.50 picks the candidate whose places of
articulation are nearest: *e* is kaṇṭha + tālu, exactly the union of *a*
(kaṇṭha) and *i* (tālu).  We define "nearest" once and **derive** the tables.
-/

namespace Ashtadhyayi
namespace Varna

/-- Places inside the mouth (āsya): nāsikā does not count for 1.1.9. -/
def oral (sv : Varna) : List Sthana := sv.sthana.filter (· ≠ .nasika)

/-- The savarṇa class of a sound: vowels and consonants are kept apart
(1.1.10 nājjhalau), ṛ and ḷ are merged (vārttika *ṛḷvarṇayor mithaḥ sāvarṇyam*). -/
def savarnaKey (sv : Varna) : Bool × List Sthana × Prayatna :=
  if sv.base = .x then (true, [.murdhan], .vivrta) else (sv.isVowel, sv.oral, sv.prayatna)

/-- **1.1.9 tulyāsyaprayatnaṃ savarṇam** (with 1.1.10 and the ṛ/ḷ vārttika). -/
def savarna (v1 v2 : Varna) : Bool := decide (savarnaKey v1 = savarnaKey v2)

end Varna

open Varna

/-- Savarṇa is an equivalence relation — by construction, since it is equality of keys. -/
theorem savarna_refl (v1 : Varna) : savarna v1 v1 = true := by simp [savarna]
theorem savarna_symm (v1 v2 : Varna) : savarna v1 v2 = savarna v2 v1 := by
  simp [savarna, eq_comm]
theorem savarna_trans (v1 v2 v3 : Varna) (h₁ : savarna v1 v2 = true) (h₂ : savarna v2 v3 = true) :
    savarna v1 v3 = true := by
  simp [savarna] at *; exact h₁.trans h₂

example : savarna .a .A = true := by decide
example : savarna .f .x = true := by decide           -- vārttika
example : savarna .e .E = false := by decide          -- vivṛtatara keeps ai apart from e
example : savarna .i .S = false := by decide          -- 1.1.10 nājjhalau
example : savarna .k .N = true := by decide           -- nasality is not an oral place

/-- 1.1.2 adeṅ guṇaḥ (at = short a by 1.1.70) and 1.1.1 vṛddhir ādaic. -/
def gunaSet : List Varna := [.a, .e, .o]
def vrddhiSet : List Varna := [.A, .E, .O]

/-- Nearness of candidate `cand` to the places `target`: more shared places
first, then fewer extra places. Encoded as one number for a simple `max`. -/
def nearness (target : List Sthana) (cand : Varna) : Nat :=
  let shared := (cand.sthana.filter (· ∈ target)).length
  let extra := (cand.sthana.filter (· ∉ target)).length
  10 * shared + (9 - extra)

/-- **1.1.50 sthāne 'ntaratamaḥ**: the candidate nearest to the originals
(for an ekādeśa, to the union of both originals' places); ties keep the
earlier candidate. -/
def antaratama (cands : List Varna) (orig : List Varna) : Varna :=
  let target := orig.flatMap Varna.sthana
  cands.foldl (fun best cand => if nearness target cand > nearness target best then cand else best)
    (cands.headD .a)

/-- **1.1.51 ur aṇ raparaḥ**: an aṇ replacing ṛ is followed by r (ḷ: by l). -/
def rapara (sub orig : Varna) : List Varna :=
  if orig.base = .f ∧ sub.base ∈ [Varna.a, .i, .u] then [sub, .r]
  else if orig.base = .x ∧ sub.base ∈ [Varna.a, .i, .u] then [sub, .l]
  else [sub]

/-- Guṇa of i, u, ṛ, ḷ — derived, not tabulated. -/
theorem guna_derived :
    (rapara (antaratama gunaSet [.i]) .i, rapara (antaratama gunaSet [.u]) .u,
     rapara (antaratama gunaSet [.f]) .f, rapara (antaratama gunaSet [.x]) .x)
    = ([.e], [.o], [.a, .r], [.a, .l]) := by decide

theorem vrddhi_derived :
    (antaratama vrddhiSet [.i], antaratama vrddhiSet [.u], rapara (antaratama vrddhiSet [.f]) .f)
    = (.E, .O, [.A, .r]) := by decide

/-- Single substitutes (6.1.84 ekaḥ pūrvaparayoḥ) come out of the same rule:
a + i → e, a + u → o, a + ṛ → ar, a + e → ai, a + o → au. -/
theorem ekadesa_derived :
    (antaratama gunaSet [.a, .i], antaratama gunaSet [.a, .u],
     rapara (antaratama gunaSet [.a, .f]) .f,
     antaratama vrddhiSet [.a, .e], antaratama vrddhiSet [.a, .o])
    = (.e, .o, [.a, .r], .E, .O) := by decide

end Ashtadhyayi
