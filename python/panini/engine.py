"""A deterministic prakriyā (derivation) engine.

Design in one paragraph
-----------------------
The utterance is a list of *segments* (one phoneme each).  Rules of the first
seven-and-a-quarter books (the *sapādasaptādhyāyī*, "phase A") run to a fixpoint;
at every step all applicable (rule, site) candidates are collected and a
*resolver* picks the winner (apavāda > antaraṅga > nitya > para).  Then the
*tripādī* (8.2–8.4) runs strictly in sūtra order.

8.2.1 pūrvatrāsiddham ("from here on, a rule is as if not effected with respect
to earlier ones") is implemented as **visibility on data**, not as a control
trick: every tripādī edit is appended to the segment's history, and a phase-A
rule reads each segment through a *view* that hides tripādī edits it cannot
see.  A rule such as 6.1.114 haśi ca, which explicitly mentions *ru* (a tripādī
output), declares ``sees={"8.2.66"}``.  After each tripādī application phase A
is simply re-run: rules that cannot see the change still see the old world and
stay quiet, rules that can see it fire.  This reproduces the textbook cases
(*hara iha* not **hareha*; *rāmo gacchati*; *harī ramate*) without special cases.

Options (vā, vibhāṣā, anyatarasyām) make the grammar a relation, not a
function.  Here every derivation is a pure function of the input and an
explicit *choice vector*; ``derive_all`` enumerates the vectors.  That is the
precise sense in which the system is deterministic.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from . import translit


# ─── data ────────────────────────────────────────────────────────────────────

@dataclass
class Seg:
    uid: int
    ph: str                                  # current phoneme; '' = deleted by a tripādī rule
    padas: frozenset[int]
    tags: frozenset[str] = frozenset()
    base: str = ""                           # value visible to phase-A rules
    base_tags: frozenset[str] = frozenset()
    hist: tuple[tuple[str, str, frozenset[str]], ...] = ()   # tripādī edits

    def __post_init__(self):
        if not self.base:
            self.base, self.base_tags = self.ph, self.tags


@dataclass
class Pada:
    text: str                                # as given by the user (SLP1)
    gender: Optional[str] = None
    derived: bool = False


@dataclass
class State:
    segs: list[Seg]
    padas: list[Pada]
    next_uid: int = 0

    def new_uid(self) -> int:
        self.next_uid += 1
        return self.next_uid - 1


@dataclass(frozen=True)
class Entry:
    """One visible segment in a rule's view."""
    idx: int                                 # index into State.segs
    ph: str
    tags: frozenset[str]
    padas: frozenset[int]
    uid: int


class View:
    def __init__(self, state: State, sees: Optional[frozenset[str]]):
        self.state = state
        self.entries: list[Entry] = []
        for i, s in enumerate(state.segs):
            if sees is None:                 # tripādī: everything already done is siddha
                ph, tags = s.ph, s.tags
            else:
                ph, tags = s.base, s.base_tags
                for rid, hph, htags in s.hist:
                    if rid not in sees:
                        break
                    ph, tags = hph, htags
            if ph:
                self.entries.append(Entry(i, ph, tags, s.padas, s.uid))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, k: int) -> Optional[Entry]:
        return self.entries[k] if 0 <= k < len(self.entries) else None

    def pada_final(self, k: int) -> bool:
        e, n = self[k], self[k + 1]
        return e is not None and (n is None or bool(e.padas - n.padas))

    def pada_initial(self, k: int) -> bool:
        e, p = self[k], self[k - 1]
        return e is not None and (p is None or bool(e.padas - p.padas))

    def avasana(self, k: int) -> bool:
        """1.4.110 virāmo 'vasānam — nothing follows."""
        return k == len(self.entries) - 1


@dataclass
class Match:
    rule: "Sutra"
    i: int                                   # first seg index replaced
    j: int                                   # last seg index replaced (inclusive)
    new: str | list[str]                     # replacement phonemes, or alternatives
    tags: frozenset[str] = frozenset()
    merge: bool = False                      # ekādeśa: new segs belong to both padas (6.1.85)
    insert: bool = False                     # augment: insert *before* seg i, replace nothing
    operand: int = 0                         # uid of the sthānin
    note: str = ""
    padas: Optional[frozenset[int]] = None   # override pada membership of new segs

    def uids(self, state: State) -> frozenset[int]:
        if self.insert:
            return frozenset({state.segs[self.i].uid})
        return frozenset(s.uid for s in state.segs[self.i:self.j + 1])


@dataclass
class Sutra:
    id: str
    text: str                                # IAST
    gloss: str
    find: Callable[[View, "Sutra"], Iterable[Match]]
    sees: frozenset[str] = frozenset()       # tripādī outputs this phase-A rule may see
    apavada_of: frozenset[str] = frozenset()
    optional: bool = False                   # vā / vibhāṣā: may be skipped
    retains: bool = False                    # applying it means "keep as is" (e.g. 8.3.36 vā śari)

    @property
    def key(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.id.split("."))

    @property
    def tripadi(self) -> bool:
        return self.key >= (8, 2, 1)

    @property
    def devanagari(self) -> str:
        return translit.slp1_to_devanagari(translit.iast_to_slp1(self.text))


@dataclass
class Step:
    sutra: str
    text: str
    gloss: str
    result: str                              # SLP1 rendering after the step
    note: str = ""


@dataclass
class Derivation:
    source: str
    result: str                              # SLP1, words separated by spaces
    steps: list[Step]
    choices: dict
    warnings: list[str] = field(default_factory=list)

    @property
    def iast(self) -> str:
        return translit.slp1_to_iast(self.result)

    @property
    def devanagari(self) -> str:
        return render_devanagari(self.result)


class NeedChoice(Exception):
    """Raised when a derivation reaches an undecided option.  ``order`` lists
    the values in preference order (for vā-rules: apply first)."""

    def __init__(self, key, order):
        super().__init__(key)
        self.key, self.order = key, list(order)


class EngineError(Exception):
    pass


# ─── editing ─────────────────────────────────────────────────────────────────

def _apply(state: State, m: Match, value: str) -> list[int]:
    """Apply match with the chosen replacement; return uids of new segments."""
    segs = state.segs
    if m.insert:
        tgt = segs[m.i]
        new = [Seg(state.new_uid(), ph, m.padas or tgt.padas, m.tags) for ph in value]
        state.segs = segs[:m.i] + new + segs[m.i:]
        return [s.uid for s in new]
    old = segs[m.i:m.j + 1]
    if m.rule.tripadi:
        # tripādī edit: recorded in history so earlier rules can ignore it
        if len(old) != 1 or len(value) > 1:
            raise EngineError(f"{m.rule.id}: tripādī edits must be 1→0/1 segments")
        s = old[0]
        s.ph, s.tags = value, (s.tags - {"ru"}) | m.tags
        s.hist = s.hist + ((m.rule.id, value, s.tags),)
        return [s.uid]
    padas = frozenset().union(*(s.padas for s in old)) if m.merge else old[0].padas
    tags = frozenset().union(*(s.base_tags for s in old)) | m.tags
    tags -= {"ru"}
    new = [Seg(state.new_uid(), ph, padas, tags) for ph in value]
    if m.merge and "rapara" in m.tags and len(new) > 1:
        # 6.1.85 antādivac ca: the r of 1.1.51 is read as initial of the
        # following word (mahā + ṛṣi → mahar-ṣi, not pada-final r).
        new[-1].padas = old[-1].padas
    state.segs = segs[:m.i] + new + segs[m.j + 1:]
    return [s.uid for s in new]


def _value(m: Match, choice: Optional[int]) -> str:
    if isinstance(m.new, list):
        return m.new[choice or 0]
    return m.new


def _is_noop(state: State, m: Match) -> bool:
    if m.insert or m.rule.retains or isinstance(m.new, list):
        return False
    old = state.segs[m.i:m.j + 1]
    cur = "".join(s.ph for s in old)
    return cur == m.new


# ─── rendering ───────────────────────────────────────────────────────────────

def render(state: State) -> str:
    out: list[str] = []
    prev: Optional[Seg] = None
    for s in state.segs:
        if not s.ph:
            continue
        if prev is not None:
            if "avagraha" in prev.tags and min(prev.padas) not in s.padas:
                out.append(" '")                     # hare + atra → hare 'tra
            elif not (prev.padas & s.padas):
                out.append(" ")
        out.append(s.ph)
        prev = s
    return "".join(out)


def render_devanagari(slp: str) -> str:
    """Join words that Devanagari cannot write apart (consonant + vowel,
    consonant + consonant); keep spaces after vowels, visarga and anusvāra."""
    words = slp.replace(" '", "'").split(" ")
    out = words[0]
    for w in words[1:]:
        last = out[-1]
        if last in "aAiIuUfFxXeEoOMH'":
            out += " " + w
        else:
            out += w
    return translit.slp1_to_devanagari(out)


# ─── conflict resolution ─────────────────────────────────────────────────────

RESOLVERS = ("classical", "sutra-order", "right-operand")

# Pairs (winner, loser) where the tradition lets the *earlier* rule win
# (pūrvavipratiṣedha, justified by Patañjali's reading of para as iṣṭa,
# "desired").  Only the case exercised by this fragment is listed.
PURVA_VIPRATISEDHA = {("7.1.9", "7.3.103")}


class Resolver:
    def __init__(self, name: str):
        if name not in RESOLVERS:
            raise ValueError(f"resolver must be one of {RESOLVERS}")
        self.name = name

    def para(self, d: Match, c: Match, state: State) -> bool:
        """Does d beat c under 1.4.2 vipratiṣedhe paraṃ kāryam?"""
        if self.name == "classical" and (d.rule.id, c.rule.id) in PURVA_VIPRATISEDHA:
            return True
        if self.name == "classical" and (c.rule.id, d.rule.id) in PURVA_VIPRATISEDHA:
            return False
        if self.name == "right-operand":
            # Rajpopat (2022): 'para' = the operation on the right-hand operand
            pd, pc = _position(state, d.operand), _position(state, c.operand)
            if pd != pc:
                return pd > pc
        return d.rule.key > c.rule.key


def _position(state: State, uid: int) -> int:
    for i, s in enumerate(state.segs):
        if s.uid == uid:
            return i
    return -1


# ─── the engine ──────────────────────────────────────────────────────────────

class Engine:
    MAX_STEPS = 400

    def __init__(self, sutras: list[Sutra], resolver: str = "classical", asiddha: bool = True):
        # asiddha=False is an ablation: every rule sees every change, as in a
        # naive rewrite system.  It exists to show why 8.2.1 is necessary.
        self.asiddha = asiddha
        self.phase_a = [s for s in sutras if not s.tripadi]
        self.tripadi = sorted((s for s in sutras if s.tripadi), key=lambda s: s.key)
        self.by_id = {s.id: s for s in sutras}
        # An exception to an exception is an exception to the rule: close the
        # apavāda relation transitively (6.1.102 ⊂ 6.1.101 ⊂ 6.1.87).
        closure = {s.id: set(s.apavada_of) for s in sutras}
        changed = True
        while changed:
            changed = False
            for a, us in closure.items():
                extra = set().union(*(closure.get(u, set()) for u in us)) - us
                if extra:
                    us |= extra
                    changed = True
        self.apavada = {a: frozenset(us) for a, us in closure.items()}
        self.apavadas: dict[str, list[Sutra]] = {}
        for s in sutras:
            for u in self.apavada[s.id]:
                self.apavadas.setdefault(u, []).append(s)
        self.resolver = Resolver(resolver)

    # candidates ------------------------------------------------------------
    def _view(self, state: State, rule: Sutra) -> View:
        if rule.tripadi or not self.asiddha:
            return View(state, None)
        return View(state, rule.sees)

    def _candidates(self, state: State, rules: list[Sutra], declined: set) -> list[Match]:
        out = []
        for r in rules:
            view = self._view(state, r)
            for m in r.find(view, r):
                if (r.id, m.operand) in declined or _is_noop(state, m):
                    continue
                out.append(m)
        return out

    def _still_applies(self, state: State, m: Match, near: frozenset[int]) -> bool:
        view = self._view(state, m.rule)
        for m2 in m.rule.find(view, m.rule):
            if not _is_noop(state, m2) and (m2.uids(state) & near):
                return True
        return False

    def _after(self, state: State, m: Match) -> tuple[State, frozenset[int]]:
        st = copy.deepcopy(state)
        new = _apply(st, m, _value(m, 0))          # indices are identical in the copy
        return st, frozenset(new)

    def _beats(self, d: Match, c: Match, state: State) -> bool:
        du, cu = d.uids(state), c.uids(state)
        if c.rule.id in self.apavada[d.rule.id] and du & cu:
            return True                                     # apavāda
        if d.rule.id in self.apavada[c.rule.id] and du & cu:
            return False
        # Conflict = one application bleeds the other.
        st_d, new_d = self._after(state, d)
        st_c, new_c = self._after(state, c)
        c_survives = self._still_applies(st_d, c, cu | new_d)
        d_survives = self._still_applies(st_c, d, du | new_c)
        if c_survives and d_survives:
            return False                                    # independent: order irrelevant
        ant_d, ant_c = len(_padas(state, d)) == 1, len(_padas(state, c)) == 1
        if ant_d != ant_c:
            return ant_d                                    # antaraṅga
        if d_survives != c_survives:
            return d_survives                               # nitya
        return self.resolver.para(d, c, state)              # para (1.4.2)

    def _choose(self, state: State, cands: list[Match]) -> Match:
        # 1. An applicable apavāda removes its utsarga from contention outright.
        live = [c for c in cands
                if not any(c.rule.id in self.apavada[d.rule.id] and c.uids(state) & d.uids(state)
                           for d in cands if d is not c)]
        if len(live) == 1:
            return live[0]
        # 2. Among the rest: antaraṅga > nitya > para (by the chosen resolver).
        wins = {id(c): sum(self._beats(c, d, state) for d in live if d is not c) for c in live}
        unbeaten = [c for c in live if not any(self._beats(d, c, state) for d in live if d is not c)]
        pool = unbeaten or sorted(live, key=lambda c: -wins[id(c)])[:1]
        return min(pool, key=lambda m: (m.i, tuple(-k for k in m.rule.key)))

    # phases ----------------------------------------------------------------
    def _phase_a(self, state, steps, choices, declined, budget):
        while True:
            cands = self._candidates(state, self.phase_a, declined)
            if not cands:
                return
            m = self._choose(state, cands)
            choice = None
            if m.rule.optional:
                if not self._choice(choices, (m.rule.id, m.operand), None):
                    declined.add((m.rule.id, m.operand))
                    continue
            if isinstance(m.new, list):
                choice = self._choice(choices, (m.rule.id, m.operand), len(m.new))
            _apply(state, m, _value(m, choice))
            steps.append(self._step(m, state))
            budget[0] += 1
            if budget[0] > self.MAX_STEPS:
                raise EngineError("derivation did not terminate")

    def _phase_b(self, state, steps, choices, declined, budget):
        for rule in self.tripadi:
            done: set[int] = set()
            while True:
                view = View(state, None)
                ms = [m for m in rule.find(view, rule)
                      if m.operand not in done and (rule.id, m.operand) not in declined
                      and not _is_noop(state, m)]
                if not ms:
                    break
                m = ms[0]
                done.add(m.operand)
                if self._suppressed(state, m, choices, declined):
                    continue
                if rule.optional and not self._choice(choices, (rule.id, m.operand), None):
                    declined.add((rule.id, m.operand))
                    continue
                choice = None
                if isinstance(m.new, list):
                    choice = self._choice(choices, (rule.id, m.operand), len(m.new))
                if rule.retains:
                    steps.append(self._step(m, state))
                    continue
                _apply(state, m, _value(m, choice))
                steps.append(self._step(m, state))
                # Re-run phase A: only rules that can *see* this edit will react.
                self._phase_a(state, steps, choices, declined, budget)

    def _suppressed(self, state, m, choices, declined) -> bool:
        """An apavāda blocks its utsarga at the same site even inside the
        ordered tripādī (8.2.66 blocks 8.2.39 though it comes later)."""
        view = View(state, None)
        uids = m.uids(state)
        for a in self.apavadas.get(m.rule.id, []):
            for am in a.find(view, a):
                if not (am.uids(state) & uids) or (a.id, am.operand) in declined:
                    continue
                if a.optional and not self._choice(choices, (a.id, am.operand), None):
                    declined.add((a.id, am.operand))
                    continue
                return True
        return False

    @staticmethod
    def _choice(choices: dict, key, n: Optional[int]) -> int:
        """n=None: a vā-rule (1 = apply, 0 = skip); else index of an alternative."""
        if key not in choices:
            raise NeedChoice(key, [1, 0] if n is None else range(n))
        return choices[key]

    @staticmethod
    def _step(m: Match, state: State) -> Step:
        return Step(m.rule.id, m.rule.text, m.rule.gloss, render(state), m.note)

    # public ----------------------------------------------------------------
    def run(self, state: State, choices: dict, steps: list[Step]) -> str:
        declined: set = set()
        budget = [0]
        self._phase_a(state, steps, choices, declined, budget)
        steps.append(Step("8.2.1", "pūrvatrāsiddham", "the tripādī begins: its rules apply in "
                          "order and are invisible to all earlier rules", render(state)))
        self._phase_b(state, steps, choices, declined, budget)
        return render(state)


def _padas(state: State, m: Match) -> frozenset[int]:
    if m.insert:
        return state.segs[m.i].padas
    return frozenset().union(*(s.padas for s in state.segs[m.i:m.j + 1]))


def enumerate_choices(run: Callable[[dict], Derivation]) -> list[Derivation]:
    """Depth-first enumeration of every choice vector (every optional branch)."""
    out: list[Derivation] = []
    stack: list[dict] = [{}]
    while stack:
        ch = stack.pop()
        try:
            out.append(run(dict(ch)))
        except NeedChoice as nc:
            for v in reversed(nc.order):
                stack.append({**ch, nc.key: v})
    return out
