"""Instances of the Fire Scheduling Problem with Common and Expandable Deadlines (FSP-CED).

An instance has m batteries (weapons) and n targets. Target j has a value w_j, a deadline
extension delta_j >= 0 (credited when j is neutralized) and a set of firing modes.  A mode is
a set of batteries that fire simultaneously at the target (synchronized salvo) together with
its duration p.  The initial (common) deadline is D0 and the deadline can never exceed the cap
Dbar.  All data are integers, time unit = 1 minute.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import numpy as np


@dataclass
class Mode:
    batteries: Tuple[int, ...]
    p: int

    @property
    def area(self) -> int:
        return self.p * len(self.batteries)


@dataclass
class Target:
    w: int
    delta: int
    modes: List[Mode]
    kind: str = ""
    x: float = 0.0
    y: float = 0.0


@dataclass
class Instance:
    name: str
    m: int
    D0: int
    Dbar: int
    targets: List[Target]
    battery_xy: List[Tuple[float, float]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.targets)

    @property
    def T(self) -> int:
        """Planning horizon (no firing can end after Dbar)."""
        return self.Dbar

    def common_variant(self) -> "Instance":
        """Same instance with all extensions removed (pure common deadline D0)."""
        tg = [Target(t.w, 0, [Mode(md.batteries, md.p) for md in t.modes], t.kind, t.x, t.y)
              for t in self.targets]
        meta = dict(self.meta)
        meta["variant"] = "CD"
        return Instance(self.name.replace("_ED", "_CD"), self.m, self.D0, self.D0, tg,
                        list(self.battery_xy), meta)

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> dict:
        return {
            "name": self.name, "m": self.m, "D0": self.D0, "Dbar": self.Dbar,
            "battery_xy": [list(b) for b in self.battery_xy],
            "meta": self.meta,
            "targets": [{"w": t.w, "delta": t.delta, "kind": t.kind, "x": t.x, "y": t.y,
                         "modes": [{"batteries": list(md.batteries), "p": md.p} for md in t.modes]}
                        for t in self.targets],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        tg = [Target(t["w"], t["delta"], [Mode(tuple(md["batteries"]), md["p"]) for md in t["modes"]],
                     t.get("kind", ""), t.get("x", 0.0), t.get("y", 0.0)) for t in d["targets"]]
        return Instance(d["name"], d["m"], d["D0"], d["Dbar"], tg,
                        [tuple(b) for b in d.get("battery_xy", [])], d.get("meta", {}))

    @staticmethod
    def load(path: str) -> "Instance":
        with open(path) as f:
            return Instance.from_dict(json.load(f))

    # ------------------------------------------------------------ helpers
    def n_modes(self) -> int:
        return sum(len(t.modes) for t in self.targets)

    def min_area(self, j: int) -> int:
        return min(md.area for md in self.targets[j].modes)


# ---------------------------------------------------------------------- dominance
def prune_dominated(modes: List[Mode]) -> List[Mode]:
    """Remove a mode if another mode uses a subset of its batteries and is not longer."""
    keep = []
    for a in modes:
        sa = set(a.batteries)
        dom = False
        for b in modes:
            if b is a:
                continue
            sb = set(b.batteries)
            if sb <= sa and b.p <= a.p and (sb != sa or b.p < a.p):
                dom = True
                break
        if not dom:
            keep.append(a)
    # remove exact duplicates
    out, seen = [], set()
    for md in keep:
        key = (tuple(sorted(md.batteries)), md.p)
        if key not in seen:
            seen.add(key)
            out.append(md)
    return out


# ---------------------------------------------------------------------- generator
TARGET_CLASSES = {
    # kind: (probability, value range, volleys range, minimum batteries, counter-battery capable,
    #        extension range as a fraction of D0)
    "HV": (0.20, (7, 10), (10, 16), 2, True, (0.10, 0.25)),   # major combatant (frigate/corvette)
    "FA": (0.30, (3, 6), (3, 6), 1, True, (0.05, 0.15)),      # fast attack / missile craft
    "LC": (0.50, (2, 8), (3, 8), 1, False, (0.0, 0.0)),       # landing craft / transport
}


def generate(m: int, n: int, tau: float, seed: int, qmax: int = 3, kmax: int = 8,
             spacing: float = 12.0, rng_range=(30.0, 40.0), cap: float = 1.5,
             name: str | None = None) -> Instance:
    """Coastal-defense scenario generator (see Appendix of the paper).

    Batteries lie along a straight coast (y = 0), about `spacing` km apart, each with a maximum
    range drawn from `rng_range`.  Targets are ships offshore.  A target can be engaged by the
    batteries within range; a mode is a set of q_min..qmax batteries that are contiguous along
    the coast among those in range.  The duration of a mode is 1 + ceil(N_j / sum_i rho_ij), where
    N_j is the number of effective volleys needed and rho_ij = 1 - 0.5 d_ij / R_i is the effective
    volley rate of battery i at distance d_ij.  D0 = round(tau * A / m), where A is the sum over
    targets of the smallest mode area p*|S|; extensions are fractions of D0 and Dbar = ceil(cap*D0).
    """
    rng = np.random.default_rng(seed)
    L = spacing * m
    bx = [((i + 0.5) * L / m + rng.uniform(-2.0, 2.0), 0.0) for i in range(m)]
    R = [float(rng.uniform(*rng_range)) for _ in range(m)]
    kinds = list(TARGET_CLASSES)
    probs = np.array([TARGET_CLASSES[k][0] for k in kinds])
    raw = []
    while len(raw) < n:
        kind = kinds[rng.choice(len(kinds), p=probs)]
        _, wr, nr, qmin, cb, er = TARGET_CLASSES[kind]
        x = float(rng.uniform(-5.0, L + 5.0))
        y = float(rng.uniform(8.0, 30.0))
        d = [math.hypot(x - bx[i][0], y - bx[i][1]) for i in range(m)]
        elig = [i for i in range(m) if d[i] <= R[i]]
        if len(elig) < qmin:
            continue  # resample: the target must be reachable by enough batteries
        w = int(rng.integers(wr[0], wr[1] + 1))
        N = int(rng.integers(nr[0], nr[1] + 1))
        efrac = float(rng.uniform(er[0], er[1])) if cb else 0.0
        rho = {i: 1.0 - 0.5 * d[i] / R[i] for i in elig}
        modes = []
        for q in range(qmin, min(qmax, len(elig)) + 1):
            for s in range(len(elig) - q + 1):
                S = tuple(elig[s:s + q])      # contiguous along the coast (batteries sorted by x)
                p = 1 + math.ceil(N / sum(rho[i] for i in S))
                modes.append(Mode(S, p))
        modes = prune_dominated(modes)
        modes.sort(key=lambda md: (md.area, md.p))
        modes = modes[:kmax]
        raw.append((kind, w, N, efrac, modes, x, y))
    A = sum(min(md.area for md in r[4]) for r in raw)
    D0 = max(1, int(round(tau * A / m)))
    targets = []
    for kind, w, N, efrac, modes, x, y in raw:
        delta = int(round(efrac * D0))
        targets.append(Target(w, delta, modes, kind, round(x, 3), round(y, 3)))
    Dbar = int(math.ceil(cap * D0))
    nm = name or f"m{m}_n{n}_t{int(round(tau * 100))}_s{seed}_ED"
    meta = {"m": m, "n": n, "tau": tau, "seed": seed, "qmax": qmax, "kmax": kmax, "cap": cap,
            "ranges": [round(r, 3) for r in R], "variant": "ED"}
    return Instance(nm, m, D0, Dbar, targets, [(round(a, 3), b) for a, b in bx], meta)


def generate_short(m: int, n: int, tau: float, seed: int, ext_prob: float = 0.6, ext_range=(1, 4),
                   name: str | None = None) -> Instance:
    """Short-window instances (Section 7 of the paper): the coastal-defense scenario of `generate` with an initial
    window of a few minutes, extensions of a few minutes and no cap.  The targets, their positions and their modes
    are those of generate(m, n, ., seed); then D0 = max(4, round(tau * A / m)) with tau around 0.05-0.12 (A is the
    sum of the smallest mode areas), a major combatant or fast attack craft extends the deadline with probability
    ext_prob by an integer number of minutes drawn uniformly from ext_range, and Dbar = D0 + sum of the extensions,
    so that the cap never binds.  With so short a window only a few engagements can be completed before the first
    extensions are needed, and the order in which the extensions are earned decides what can be engaged."""
    base = generate(m, n, 0.3, seed)
    rng = np.random.default_rng(seed + 1_000_003)
    A = sum(min(md.area for md in t.modes) for t in base.targets)
    D0 = max(4, int(round(tau * A / m)))
    targets = []
    for t in base.targets:
        d = 0
        if t.kind in ("HV", "FA") and rng.random() < ext_prob:
            d = int(rng.integers(ext_range[0], ext_range[1] + 1))
        targets.append(Target(t.w, d, [Mode(md.batteries, md.p) for md in t.modes], t.kind, t.x, t.y))
    Dbar = D0 + sum(t.delta for t in targets)
    nm = name or f"sw_m{m}_n{n}_t{int(round(tau * 100)):02d}_s{seed}_ED"
    meta = dict(base.meta)
    meta.update({"family": "short-window", "tau": tau, "ext_prob": ext_prob, "ext_range": list(ext_range),
                 "cap": None, "variant": "ED"})
    return Instance(nm, m, D0, Dbar, targets, list(base.battery_xy), meta)


def generate_dispersed(m: int, n: int, tau: float, seed: int, short: bool = False, reach=(2, 4), qmax: int = 3,
                       kmax: int = 8, cap: float = 1.5, ext_prob: float = 0.6, ext_range=(1, 4),
                       name: str | None = None) -> Instance:
    """Dispersed batteries (Section 7 of the paper): the target classes of the coastal generator without its
    geometry, so that the battery sets of the modes are arbitrary.  Target j is within range of a random set of
    batteries, whose size is uniform in reach (and at least the minimum group of its class); battery i fires
    r_ij ~ U(0.5, 1) effective volleys per minute at it; and a mode is ANY group of q_min..qmax batteries within range,
    with duration 1 + ceil(N_j / sum_i r_ij).  Dominated modes are removed and the kmax modes of smallest battery time
    are kept.  Conflict families without a common battery (e.g. the pairs of three batteries) are therefore frequent,
    and the battery loads no longer decide whether a selection can be packed.
    Two window regimes:
      short = False  D0 = round(tau * A / m), extensions as fractions of D0 by class, Dbar = ceil(cap * D0);
      short = True   D0 = max(4, round(tau * A / m)), a major combatant or fast attack craft extends the deadline with
                     probability ext_prob by ext_range minutes, and Dbar = D0 + sum of the extensions (no cap)."""
    rng = np.random.default_rng(seed)
    kinds = list(TARGET_CLASSES)
    probs = np.array([TARGET_CLASSES[k][0] for k in kinds])
    raw = []
    for _ in range(n):
        kind = kinds[rng.choice(len(kinds), p=probs)]
        _, wr, nr, qmin, cb, er = TARGET_CLASSES[kind]
        k = int(rng.integers(max(reach[0], qmin), min(reach[1], m) + 1))
        elig = sorted(int(i) for i in rng.choice(m, k, replace=False))
        rate = {i: float(rng.uniform(0.5, 1.0)) for i in elig}
        w = int(rng.integers(wr[0], wr[1] + 1))
        N = int(rng.integers(nr[0], nr[1] + 1))
        efrac = float(rng.uniform(er[0], er[1])) if cb else 0.0
        gives = bool(cb and rng.random() < ext_prob)
        minutes = int(rng.integers(ext_range[0], ext_range[1] + 1))
        modes = [Mode(S, 1 + math.ceil(N / sum(rate[i] for i in S)))
                 for q in range(qmin, min(qmax, k) + 1) for S in itertools.combinations(elig, q)]
        modes = prune_dominated(modes)
        modes.sort(key=lambda md: (md.area, md.p, md.batteries))
        raw.append((kind, w, efrac, gives, minutes, modes[:kmax]))
    A = sum(min(md.area for md in r[5]) for r in raw)
    if short:
        D0 = max(4, int(round(tau * A / m)))
        targets = [Target(w, minutes if gives else 0, modes, kind) for kind, w, _, gives, minutes, modes in raw]
        Dbar = D0 + sum(t.delta for t in targets)
    else:
        D0 = max(1, int(round(tau * A / m)))
        targets = [Target(w, int(round(efrac * D0)), modes, kind) for kind, w, efrac, _, _, modes in raw]
        Dbar = int(math.ceil(cap * D0))
    nm = name or f"dp_m{m}_n{n}_{'s' if short else 'c'}{int(round(tau * 100)):02d}_s{seed}_ED"
    meta = {"family": "dispersed", "m": m, "n": n, "tau": tau, "seed": seed, "short": bool(short),
            "reach": list(reach), "qmax": qmax, "kmax": kmax, "cap": None if short else cap, "variant": "ED"}
    return Instance(nm, m, D0, Dbar, targets, [], meta)


def optima_example() -> Instance:
    """Illustrative instance of Canut-de-Bon et al. (2017): 3 identical batteries, 5 targets,
    common deadline 41.  Modes enumerate all battery subsets of the required size."""
    data = [(1, 15, 10), (2, 10, 3), (2, 13, 7), (1, 20, 9), (3, 18, 15)]  # (W_j, p_j, w_j)
    tg = []
    for W, p, w in data:
        modes = [Mode(S, p) for S in itertools.combinations(range(3), W)]
        tg.append(Target(w, 0, modes, "", 0.0, 0.0))
    return Instance("optima2017", 3, 41, 41, tg, [], {"source": "Canut-de-Bon et al. (2017)"})


def generate_chakim(m: int, n: int, single_pct: int, seed: int, tau: float = 0.4, cap: float = 1.5,
                    name: str | None = None) -> Instance:
    """Instances following the published design of Cha and Kim (2010, Omega 38:383-392), adapted to the
    FSP-CED.  As in their study, every target has pre-assigned weapons: a single-weapon target (share
    single_pct %) has one random weapon and a firing duration in {2, 3}; a multiple-weapon target has 2 or 3
    random weapons (equal probability), one of which fires for 1 period and the others for {2, 3}; weapon i
    has a set-up time s_i in U{1..4}, and the threat value of target j is w_j in U{2..100}.  Because the
    batteries of a salvo start together and are released together in the FSP-CED, the single mode of
    target j lasts p_j = max_{i in W_j} (s_i + d_ij).  Additions for the FSP-CED: D0 = round(tau * sum_j
    p_j |W_j| / m); multiple-weapon targets extend the deadline with probability 0.5 and single-weapon
    targets with probability 0.2, by delta_j = max(1, round(U(0.05, 0.20) * D0)); Dbar = ceil(cap * D0)."""
    rng = np.random.default_rng(seed)
    setup = [int(rng.integers(1, 5)) for _ in range(m)]
    n_single = int(round(single_pct / 100.0 * n))
    kinds = ["S"] * n_single + ["M"] * (n - n_single)
    rng.shuffle(kinds)
    raw = []
    for kind in kinds:
        w = int(rng.integers(2, 101))
        if kind == "S":
            W = [int(rng.integers(m))]
            d = {W[0]: int(rng.integers(2, 4))}
        else:
            q = min(m, int(rng.choice([2, 3])))
            W = sorted(int(i) for i in rng.choice(m, q, replace=False))
            one = W[int(rng.integers(q))]
            d = {i: (1 if i == one else int(rng.integers(2, 4))) for i in W}
        p = max(setup[i] + d[i] for i in W)
        ext = (rng.random() < (0.5 if kind == "M" else 0.2))
        frac = float(rng.uniform(0.05, 0.20)) if ext else 0.0
        raw.append((kind, w, tuple(W), p, frac))
    D0 = max(1, int(round(tau * sum(p * len(W) for _, _, W, p, _ in raw) / m)))
    targets = [Target(w, (max(1, int(round(f * D0))) if f > 0 else 0), [Mode(W, p)],
                      "CK-" + kind) for kind, w, W, p, f in raw]
    nm = name or f"ck_m{m}_n{n}_p{single_pct}_s{seed}_ED"
    meta = {"family": "Cha-Kim", "m": m, "n": n, "single_pct": single_pct, "seed": seed, "tau": tau,
            "cap": cap, "setup": setup, "variant": "ED"}
    return Instance(nm, m, D0, int(math.ceil(cap * D0)), targets, [], meta)
