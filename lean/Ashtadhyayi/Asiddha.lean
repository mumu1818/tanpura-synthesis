import Ashtadhyayi.Sandhi

/-!
# 8.2.1 *pūrvatrāsiddham* is necessary

A two-phase model: phase A is the vowel sandhi of `Sandhi.lean`, iterated to a
fixpoint; phase B ("tripādī") is 8.3.19 *lopaḥ śākalyasya* — after a/ā, a
word-final y or v before a vowel is optionally dropped.

Pāṇini makes phase-B results **asiddha** (invisible) to phase A.  We prove that
the faithful model yields exactly *harayiha* and *hara iha*, while a naive
rewrite system in which phase A can see phase-B output also produces the
ungrammatical \*hareha — so the metarule is doing real work.
-/

namespace Ashtadhyayi

/-- One vowel-sandhi step at the leftmost hiatus. -/
def sandhiStep : List Varna → Option (List Varna)
  | v1 :: v2 :: rest =>
      if v1.isVowel && v2.isVowel then some (acSandhi v1 v2 ++ rest)
      else (sandhiStep (v2 :: rest)).map (v1 :: ·)
  | _ => none

def phaseA : Nat → List Varna → List Varna
  | 0, word => word
  | fuel + 1, word => match sandhiStep word with
    | some word' => phaseA fuel word'
    | none => word

/-- 8.3.19: the (first) y/v after a/ā and before a vowel, deleted. -/
def lopa : List Varna → Option (List Varna)
  | v1 :: mid :: v2 :: rest =>
      if avarna v1 && (mid == .y || mid == .v) && v2.isVowel then some (v1 :: v2 :: rest)
      else (lopa (mid :: v2 :: rest)).map (v1 :: ·)
  | _ => none

/-- Phase B with its option: without and with the deletion. -/
def tripadi (word : List Varna) : List (List Varna) :=
  match lopa word with
  | some word' => [word, word']
  | none => [word]

/-- Faithful model: phase A runs to a fixpoint, then the tripādī; nothing
the tripādī does is ever handed back to phase A. -/
def derivations (word : List Varna) : List (List Varna) := tripadi (phaseA word.length word)

/-- Naive model: phase A may see (and rewrite) tripādī output. -/
def naiveDerivations (word : List Varna) : List (List Varna) :=
  (tripadi (phaseA word.length word)).map fun word' => phaseA word'.length word'

def hare_iha : List Varna := [.h, .a, .r, .e, .i, .h, .a]

theorem asiddha_hare_iha :
    derivations hare_iha = [[.h, .a, .r, .a, .y, .i, .h, .a],     -- harayiha
                            [.h, .a, .r, .a, .i, .h, .a]] := by   -- hara iha
  decide

/-- Without 8.2.1 the grammar overgenerates \*hareha. -/
theorem naive_overgenerates : [.h, .a, .r, .e, .h, .a] ∈ naiveDerivations hare_iha := by
  decide

/-- The same for viṣṇo + iha → viṣṇav iha / viṣṇa iha (never \*viṣṇeha). -/
theorem asiddha_visno_iha :
    derivations [.v, .i, .z, .R, .o, .i, .h, .a] =
      [[.v, .i, .z, .R, .a, .v, .i, .h, .a], [.v, .i, .z, .R, .a, .i, .h, .a]] ∧
    [.v, .i, .z, .R, .e, .h, .a] ∈ naiveDerivations [.v, .i, .z, .R, .o, .i, .h, .a] := by
  decide

end Ashtadhyayi
