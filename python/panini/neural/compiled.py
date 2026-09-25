"""Compile sūtras into an *exact* ReLU network ("Pāṇini-compiled MLP").

Idea
----
Place every sound on a line in Śiva-sūtra order.  Because every pratyāhāra is
an interval of that line (1.1.71), membership in a pratyāhāra is

    [lo ≤ p ≤ hi]  =  step(p − lo) − step(p − hi − 1),
    step(z) = ReLU(z + 1) − ReLU(z)          (exact for integer z)

i.e. a constant number of ReLU units per class, however many sounds the class
contains.  A sūtra is then an AND of such tests (one more ReLU), an apavāda is
an inhibitory connection from the exception's unit to the general rule's unit,
and the substitute is a gated table lookup.  The result is an ordinary
feed-forward ReLU network — loadable into any DL framework, usable frozen as a
guaranteed-correct module or as an interpretable initialisation — that
computes the vowel-sandhi of 6.1.77/78/87/88/101 *exactly*.

The tables (yaṇ of i, guṇa of a+i, …) are not written by hand: they are
generated from `panini.samjna` (sthāne 'ntaratamaḥ), i.e. from the same code
the symbolic engine uses, and `verify()` checks the network against the
engine on every vowel pair.

This file needs numpy.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import numpy as np

from .. import samjna, varna
from ..shivasutra import SOUNDS, members, pratyahara

BASE_VOWELS = [s for s in SOUNDS if varna.is_vowel(s)]          # a i u ṛ ḷ e o ai au
ALL_VOWELS = "aAiIuUfFxXeEoO"


def embed(v: str) -> tuple[float, float, float]:
    """Śiva-sūtra embedding of a vowel: (position, savarṇa class, length)."""
    b = varna.base_vowel(v)
    p = BASE_VOWELS.index(b)
    c = BASE_VOWELS.index("f") if b == "x" else p                # ṛ/ḷ vārttika
    return float(p), float(c), float(varna.get(v).matra == 2 and v not in "eEoO")


def interval(name: str) -> tuple[int, int]:
    ps = [BASE_VOWELS.index(s) for s in pratyahara(name) if varna.is_vowel(s)]
    lo, hi = min(ps), max(ps)
    assert ps == list(range(lo, hi + 1)), f"{name} is not an interval"
    return lo, hi


# ─── a tiny graph → layered-MLP compiler ─────────────────────────────────────

@dataclass
class Unit:
    deps: dict[str, float]
    bias: float
    depth: int
    relu: bool = True


class Circuit:
    """Build a DAG of ReLU units by name; `compile()` lays it out as dense
    layers, inserting identity pass-throughs (valid because every value is ≥ 0)."""

    def __init__(self, inputs: list[str]):
        self.inputs = list(inputs)
        self.units: dict[str, Unit] = {n: Unit({}, 0.0, 0) for n in inputs}
        self.outputs: list[str] = []

    def add(self, name: str, deps: dict[str, float], bias: float = 0.0) -> str:
        assert name not in self.units, name
        depth = 1 + max(self.units[d].depth for d in deps)
        self.units[name] = Unit(dict(deps), bias, depth)
        return name

    def output(self, name: str, deps: dict[str, float], bias: float = 0.0) -> str:
        self.units[name] = Unit(dict(deps), bias, -1, relu=False)
        self.outputs.append(name)
        return name

    def compile(self) -> "MLP":
        D = max(u.depth for u in self.units.values() if u.relu)
        for o in self.outputs:
            self.units[o].depth = D + 1
        need_until: dict[str, int] = {}
        for n, u in self.units.items():
            for d in u.deps:
                need_until[d] = max(need_until.get(d, 0), u.depth - 1)
        layers: list[list[str]] = [self.inputs]
        for d in range(1, D + 2):
            names = [n for n, u in self.units.items() if u.depth == d]
            if d <= D:
                names += [f"{n}" for n, u in self.units.items()
                          if u.depth < d and need_until.get(n, 0) >= d]
            layers.append(names)
        Ws, bs = [], []
        for d in range(1, D + 2):
            prev, cur = layers[d - 1], layers[d]
            idx = {n: i for i, n in enumerate(prev)}
            W = np.zeros((len(cur), len(prev)))
            b = np.zeros(len(cur))
            for r, n in enumerate(cur):
                u = self.units[n]
                if u.depth == d:
                    for dep, w in u.deps.items():
                        W[r, idx[dep]] += w
                    b[r] = u.bias
                else:                                           # pass-through
                    W[r, idx[n]] = 1.0
            Ws.append(W)
            bs.append(b)
        return MLP(Ws, bs, layers)


@dataclass
class MLP:
    Ws: list[np.ndarray]
    bs: list[np.ndarray]
    layers: list[list[str]]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            h = W @ h + b
            if i < len(self.Ws) - 1:
                h = np.maximum(h, 0.0)
        return h

    @property
    def shape(self) -> list[int]:
        return [len(l) for l in self.layers]

    @property
    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.Ws, self.bs))

    @property
    def n_nonzero(self) -> int:
        return sum(int((W != 0).sum() + (b != 0).sum()) for W, b in zip(self.Ws, self.bs))


# ─── the sūtras, declaratively ───────────────────────────────────────────────

@dataclass
class RuleSpec:
    id: str
    conds: list[tuple[str, str]]              # (operand, pratyāhāra | "avarna") ; ("xy","savarna")
    left: tuple[str, dict[str, str]]          # output replacing x: (source operand, base-vowel → string)
    keep_right: bool                          # does y survive? (False for ekādeśa)
    apavada_of: list[str] = field(default_factory=list)


def rule_specs() -> list[RuleSpec]:
    yan = {b: samjna.antaratama(pratyahara("yaR"), [b]) for b in "iufx"}
    ayadi = {"e": "ay", "o": "av", "E": "Ay", "O": "Av"}
    guna = {b: samjna.rapara(samjna.antaratama(samjna.GUNA, ["a", b]), b) for b in BASE_VOWELS}
    vrd = {b: samjna.antaratama(samjna.VRDDHI, ["a", b]) for b in "eoEO"}
    dirgha = {b: samjna.dirgha(b) for b in "aiufx"}
    return [
        RuleSpec("6.1.77", [("x", "ik"), ("y", "ac")], ("x", yan), True),
        RuleSpec("6.1.78", [("x", "ec"), ("y", "ac")], ("x", ayadi), True),
        RuleSpec("6.1.87", [("x", "avarna"), ("y", "ac")], ("y", guna), False),
        RuleSpec("6.1.88", [("x", "avarna"), ("y", "ec")], ("y", vrd), False, ["6.1.87"]),
        RuleSpec("6.1.101", [("x", "ak"), ("y", "ac"), ("xy", "savarna")], ("x", dirgha), False,
                 ["6.1.77", "6.1.87"]),
    ]


@dataclass
class CompiledSandhi:
    mlp: MLP
    left_vocab: list[str]
    right_vocab: list[str]

    def __call__(self, x: str, y: str) -> str:
        out = self.mlp(np.array(embed(x) + embed(y)))
        nl = len(self.left_vocab)
        l = self.left_vocab[int(np.argmax(out[:nl]))]
        r = self.right_vocab[int(np.argmax(out[nl:]))]
        return l + r


def compile_ac_sandhi() -> CompiledSandhi:
    C = Circuit(["px", "cx", "Lx", "py", "cy", "Ly"])
    # layer 1: step units on the Śiva-sūtra axis
    step_cache: dict[tuple[str, int], tuple[str, str]] = {}

    def step(op: str, t: int) -> dict[str, float]:
        """Linear expression equal to [p_op ≥ t] (for integer p)."""
        key = (op, t)
        if key not in step_cache:
            a = C.add(f"s1_{op}{t}", {f"p{op}": 1.0}, -t + 1.0)
            b = C.add(f"s0_{op}{t}", {f"p{op}": 1.0}, -float(t))
            step_cache[key] = (a, b)
        a, b = step_cache[key]
        return {a: 1.0, b: -1.0}

    def member(op: str, lo: int, hi: int) -> str:
        name = f"in_{op}_{lo}_{hi}"
        if name not in C.units:
            expr = step(op, lo)
            for k, w in step(op, hi + 1).items():
                expr[k] = expr.get(k, 0.0) - w
            C.add(name, expr)
        return name

    # savarṇa: |c_x − c_y| = 0
    dpos = C.add("d+", {"cx": 1.0, "cy": -1.0})
    dneg = C.add("d-", {"cx": -1.0, "cy": 1.0})
    sav = C.add("savarna", {dpos: -1.0, dneg: -1.0}, 1.0)

    def cond_unit(op: str, what: str) -> str:
        if what == "savarna":
            return sav
        lo, hi = (0, 0) if what == "avarna" else interval(what)
        return member(op, lo, hi)

    # one-hot of each operand's base vowel and of its full identity (with length)
    onehot = {op: {b: member(op, i, i) for i, b in enumerate(BASE_VOWELS)} for op in "xy"}
    full = {}
    for v in ALL_VOWELS:
        b = varna.base_vowel(v)
        long_ = embed(v)[2] == 1.0
        full[v] = C.add(f"is_y_{v}", {onehot["y"][b]: 1.0, "Ly": 1.0 if long_ else -1.0},
                        -1.0 if long_ else 0.0)

    # sūtra units (AND of conditions), then apavāda inhibition
    specs = rule_specs()
    app = {}
    for r in specs:
        conds = [cond_unit(op, what) for op, what in r.conds]
        app[r.id] = C.add(f"app_{r.id}", {c: 1.0 for c in conds}, -(len(conds) - 1.0))
    fired = {}
    for r in specs:
        inhibitors = [s.id for s in specs if r.id in s.apavada_of]
        fired[r.id] = C.add(f"fire_{r.id}", {app[r.id]: 1.0, **{app[i]: -1.0 for i in inhibitors}})

    # gated lookups → output logits
    left_vocab = sorted({s for r in specs for s in r.left[1].values()})
    right_vocab = [""] + list(ALL_VOWELS)
    left_terms: dict[str, dict[str, float]] = {t: {} for t in left_vocab}
    right_terms: dict[str, dict[str, float]] = {t: {} for t in right_vocab}
    for r in specs:
        src, table = r.left
        for b, tok in table.items():
            g = C.add(f"g_{r.id}_{b}", {fired[r.id]: 1.0, onehot[src][b]: 1.0}, -1.0)
            left_terms[tok][g] = 1.0
        if r.keep_right:
            for v in ALL_VOWELS:
                g = C.add(f"k_{r.id}_{v}", {fired[r.id]: 1.0, full[v]: 1.0}, -1.0)
                right_terms[v][g] = 1.0
        else:
            right_terms[""][fired[r.id]] = 1.0
    for t in left_vocab:
        C.output(f"L[{t}]", left_terms[t])
    for t in right_vocab:
        C.output(f"R[{t}]", right_terms[t])
    return CompiledSandhi(C.compile(), left_vocab, right_vocab)


# ─── verification against the symbolic engine ────────────────────────────────

def symbolic_ac_sandhi(x: str, y: str) -> str:
    from ..engine import Engine, Pada, Seg, State
    from ..sutras import BY_ID
    rules = [BY_ID[i] for i in ("6.1.77", "6.1.78", "6.1.87", "6.1.88", "6.1.101")]
    st = State([], [Pada(x + y)])
    for ph in (x, y):
        st.segs.append(Seg(st.new_uid(), ph, frozenset({0})))
    eng = Engine(rules)
    eng._phase_a(st, [], {}, set(), [0])
    return "".join(s.ph for s in st.segs)


def verify(net: CompiledSandhi | None = None) -> tuple[int, int, list[tuple[str, str, str, str]]]:
    net = net or compile_ac_sandhi()
    bad = []
    pairs = list(itertools.product(ALL_VOWELS, ALL_VOWELS))
    for x, y in pairs:
        want, got = symbolic_ac_sandhi(x, y), net(x, y)
        if want != got:
            bad.append((x, y, want, got))
    return len(pairs) - len(bad), len(pairs), bad


# ─── why the Śiva-sūtra order matters to a network ──────────────────────────

def runs(order: list[str], cls: frozenset[str]) -> int:
    """Number of maximal runs of class members along `order` (duplicates allowed:
    a duplicated sound may be left out of a run if another copy is covered)."""
    best = None
    dup = {s for s in order if order.count(s) > 1}
    # try every way of choosing which copies of duplicated members to use
    choices = [[i for i, s in enumerate(order) if s == d] for d in sorted(dup & cls)]
    for pick in itertools.product(*choices) if choices else [()]:
        use = {i for i, s in enumerate(order) if s in cls and s not in dup} | set(pick)
        n = sum(1 for i in sorted(use) if i - 1 not in use)
        best = n if best is None else min(best, n)
    return best or 0


def threshold_cost(order: list[str], classes: list[frozenset[str]]) -> int:
    """ReLU units needed to test every class on a scalar embedding: 4 per run."""
    return sum(4 * runs(order, c) for c in classes)


def sivasutra_vs_random(n_random: int = 200, seed: int = 0) -> dict:
    from ..shivasutra import USED
    classes = [frozenset(pratyahara(n)) for n in USED]
    siva = threshold_cost(SOUNDS, classes)
    rng = random.Random(seed)
    uniq = sorted(set(SOUNDS))
    costs = []
    for _ in range(n_random):
        o = uniq[:]
        rng.shuffle(o)
        costs.append(threshold_cost(o, classes))
    one_hot = sum(len(c) for c in classes)
    return {"classes": len(classes), "sivasutra_units": siva,
            "random_mean_units": sum(costs) / len(costs), "random_min_units": min(costs),
            "one_hot_linear_terms": one_hot}
