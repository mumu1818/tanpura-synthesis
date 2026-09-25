# Aṣṭādhyāyī-as-code: deterministic Sanskrit from Pāṇini's grammar

A research branch on encoding Pāṇini's *Aṣṭādhyāyī* **exactly**:

- as a traceable derivation engine (Python),
- as machine-checked mathematics (Lean 4),
- as neural networks that either *are* the grammar (compiled into their
  weights) or can only choose among moves the grammar licenses,
- and as a speech synthesiser whose every parameter comes from Pāṇinian
  phonetics.

> This branch is intentionally unrelated to the rest of the repository: it is
> an orphan branch with its own history.

**Start here → [`docs/RESEARCH.md`](docs/RESEARCH.md)**. It covers why the
Aṣṭādhyāyī is a formal system, what exists, what is proved here, the proposed
neural architectures, and a phased roadmap.

## What is in the box

| Path | What | Status |
|---|---|---|
| `python/panini/` | prakriyā engine: 32 operational sūtras + saṃjñās/metarules, sūtra-cited traces, three readings of 1.4.2, asiddha as data visibility, options as an explicit choice vector | 24 tests pass |
| `python/panini/neural/compiled.py` | compiles sūtras into an **exact ReLU MLP** on a Śiva-sūtra embedding | 196/196 vowel pairs match the engine |
| `python/panini/neural/env.py` | gym-style environment whose action space = licensed sūtra applications | interface for the policy network |
| `python/panini/speech/` | deterministic speech: durations 1.2.27, pitch 1.2.29–32, spectra from sthāna/prayatna | reproducible WAV output |
| `lean/` | Lean 4 formalization: Śiva-sūtras, 1.1.71, savarṇa, 1.1.50, vowel sandhi, asiddha, minimality | 30 theorems, no `sorry` |

## Quick start

```bash
cd python
python -m panini derive "rāma+su gacchati"      # full prakriyā with sūtra citations
python -m panini forms  "hare iha"              # every licensed output
python -m panini decline rāma
python -m panini resolve "rāma+bhis"            # competing theories of 1.4.2
python -m panini asiddha "hare iha"             # with vs. without 8.2.1
python -m panini compile                        # grammar → exact neural network, verified
python -m panini speak "rāmo gacchati" rama.wav # needs numpy
python -m unittest discover -s tests

cd ../lean && lake build                        # checks all proofs (Lean 4.22, no Mathlib)
```

Input may be IAST, SLP1 or Devanagari. `stem+ending` derives a nominal form
(masculine a-stems: `rāma+su`, `rāma+bhis`, `rāma+2.3`, …). Other words are
treated as finished padas.

```text
$ python -m panini derive "rāma+su gacchati"
1.2.45   arthavad adhātur apratyayaḥ prāti… → rāma   (rāma is a prātipadika)
4.1.2    svaujasamauṭchaṣṭābhyāmbhisṅebhyā… → rāma + su~   (vibhakti 1, vacana 1: su)
1.3.2    upadeśe 'j anunāsika it            → rāma + su~   ('u' is it)
1.3.9    tasya lopaḥ                        → rāmas   (it-markers deleted)
1.4.14   suptiṅantaṃ padam                  → rāmas   (rāmas is a pada)
8.2.1    pūrvatrāsiddham                    → rāmas gacchati
8.2.66   sasajuṣo ruḥ                       → rāmar gacchati   (pada-final s → ru (u is it, 1.3.2))
6.1.114  haśi ca                            → rāmau gacchati   (ru after a, before the haś g)
6.1.87   ād guṇaḥ                           → rāmo gacchati   (one guṇa for both (6.1.84); nearest by 1.1.50 is o)
         result                             = rāmo gacchati   रामो गच्छति

$ python -m panini resolve "rāma+bhis"
classical      rāmaiḥ
sutra-order    rāmebhiḥ
right-operand  rāmaiḥ

$ python -m panini asiddha "hare iha"
with 8.2.1   : hara iha, haray iha
without 8.2.1: hareha, haray iha
```

Notes on the samples:

- 6.1.114 fires *after* the tripādī rule 8.2.66, because it names *ru*
  explicitly and is therefore allowed to see it.
- `resolve` compares readings of 1.4.2:
  - naive "later sūtra wins" gives \*rāmebhiḥ (a Vedic, not classical, form);
  - Rajpopat's (2022) "right-hand operand wins" gives *rāmaiḥ*;
  - the classical tradition also reaches *rāmaiḥ*, but only through a
    *pūrvavipratiṣedha* exception.
- `asiddha` shows that without 8.2.1 the engine produces \*hareha.

## Highlights

* **Nothing is a lookup table.** Guṇa of *i* is *e*, and *a + i → e*, because
  1.1.50 *sthāne 'ntaratamaḥ* picks the candidate whose places of
  articulation are nearest. This is proved in Lean (`guna_derived`,
  `ekadesa_derived`) and used by the engine.
* **Asiddha (8.2.1) as data visibility.** Tripādī edits are hidden from
  earlier rules unless a rule explicitly names them (6.1.114 names *ru*).
  Removing the mechanism reproduces the classic errors.
* **Why the Śiva-sūtras are clever, quantitatively.** On Pāṇini's ordering,
  every class test costs 4 ReLU units: 168 units for all 42 pratyāhāras,
  versus ~1,093 for random orders. Lean proves that the repetition of *h* is
  forced.
* **Deterministic ≠ unique.** Optional rules make the grammar a relation.
  Here every derivation is a pure function of the input and an explicit
  choice vector, and `derive_all` enumerates them.

## Requirements

Python ≥ 3.10 (the engine is standard library only; numpy is needed for
`neural.compiled` and `speech`). Lean via [elan](https://github.com/leanprover/elan).
