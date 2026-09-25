"""A gym-style environment for neuro-symbolic prakriyā.

The observation is a derivation in progress; the *legal actions* are exactly
the sūtra applications the grammar licenses at that point.  A policy — neural
or otherwise — only chooses **which licensed rule fires next**, so every
trajectory is a valid Pāṇinian derivation and every output is grammatical by
construction.  What the policy learns is what the grammar leaves to
meta-rules (vipratiṣedha, nitya, antaraṅga…) and to the speaker (options).

This is the interface for the "Prakriyā policy network" in docs/RESEARCH.md §5.2.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..engine import Engine, Match, render
from ..prakriya import build
from ..sutras import SUTRAS


@dataclass
class Action:
    sutra: str
    text: str
    note: str
    features: dict[str, float]


class PrakriyaEnv:
    def __init__(self, text: str, *, resolver: str = "classical", filter_apavada: bool = True):
        self.text = text
        self.engine = Engine(SUTRAS, resolver=resolver)
        self.filter_apavada = filter_apavada
        self.reset()

    def reset(self) -> str:
        self.state, self.steps, self.warnings = build(self.text)
        self.trajectory: list[str] = []
        return render(self.state)

    # --- actions -----------------------------------------------------------
    def _matches(self) -> list[Match]:
        cands = self.engine._candidates(self.state, self.engine.phase_a, set())
        if self.filter_apavada:
            ap = self.engine.apavada
            cands = [c for c in cands
                     if not any(c.rule.id in ap[d.rule.id] and c.uids(self.state) & d.uids(self.state)
                                for d in cands if d is not c)]
        return cands

    def legal_actions(self) -> list[Action]:
        ms = self._matches()
        n = max(len(self.state.segs) - 1, 1)
        out = []
        for m in ms:
            pos = next(i for i, s in enumerate(self.state.segs) if s.uid == m.operand)
            padas = {p for s in self.state.segs[m.i:m.j + 1] for p in s.padas}
            out.append(Action(m.rule.id, m.rule.text, m.note, {
                "adhyaya": m.rule.key[0], "sutra_rank": sum(k * 1000 ** (2 - i) for i, k in enumerate(m.rule.key)),
                "operand_position": pos / n, "internal": float(len(padas) == 1),
                "ekadesa": float(m.merge),
            }))
        return out

    def oracle(self) -> int:
        """Index of the action the engine's own resolver would take."""
        ms = self._matches()
        best = self.engine._choose(self.state, ms)
        return ms.index(best)

    def step(self, a: int) -> tuple[str, bool]:
        ms = self._matches()
        from ..engine import _apply, _value
        m = ms[a]
        _apply(self.state, m, _value(m, 0))
        self.trajectory.append(m.rule.id)
        return render(self.state), not self._matches()

    def finish(self, choices: dict | None = None) -> str:
        """Run the (strictly ordered) tripādī and return the surface form."""
        st = copy.deepcopy(self.state)
        steps: list = []
        eng = self.engine
        ch = choices if choices is not None else _AlwaysFirst()
        eng._phase_b(st, steps, ch, set(), [0])
        return render(st)


class _AlwaysFirst(dict):
    """Choice vector that takes the preferred option everywhere."""

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return 1 if key[0] in {s.id for s in SUTRAS if s.optional} else 0


def reachable(text: str, **kw) -> dict[str, list[list[str]]]:
    """Exhaustively explore the licensed action space: surface form → the rule
    sequences (phase A) that produce it."""
    out: dict[str, list[list[str]]] = {}

    def go(env: PrakriyaEnv):
        acts = env.legal_actions()
        if not acts:
            out.setdefault(env.finish(), []).append(list(env.trajectory))
            return
        for a in range(len(acts)):
            e2 = copy.deepcopy(env)
            e2.step(a)
            go(e2)

    go(PrakriyaEnv(text, **kw))
    return out
