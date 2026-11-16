"""Single-battery FSP-CED (Theorem 2): Johnson's rule decides feasibility of a target set and a
dynamic program over Johnson's order solves the problem in O(n * Dbar * (Dbar - D0)) time."""
from __future__ import annotations

from .instance import Instance


def johnson_order(p, d, idx):
    first = sorted([j for j in idx if p[j] <= d[j]], key=lambda j: p[j])
    last = sorted([j for j in idx if p[j] > d[j]], key=lambda j: -d[j])
    return first + last


def feasible_set(inst: Instance, S) -> bool:
    p = [t.modes[0].p for t in inst.targets]
    d = [t.delta for t in inst.targets]
    P = E = 0
    for j in johnson_order(p, d, S):
        P += p[j]
        if P > min(inst.Dbar, inst.D0 + E):
            return False
        E += d[j]
    return True


def solve_dp(inst: Instance):
    """Exact DP for m = 1 (all modes use battery 0; each target has a single mode)."""
    assert inst.m == 1 and all(len(t.modes) == 1 for t in inst.targets)
    p = [t.modes[0].p for t in inst.targets]
    d = [t.delta for t in inst.targets]
    w = [t.w for t in inst.targets]
    D0, Db = inst.D0, inst.Dbar
    Emax = max(0, Db - D0)
    order = johnson_order(p, d, range(inst.n))
    # f[(P, E)] = best value with total duration P and (capped) credit E; keep parent pointers
    f = {(0, 0): (0, None, None)}
    for j in order:
        g = dict(f)
        for (P, E), (v, _, _) in f.items():
            P2 = P + p[j]
            if P2 <= min(Db, D0 + E):
                E2 = min(Emax, E + d[j])
                if (P2, E2) not in g or g[(P2, E2)][0] < v + w[j]:
                    g[(P2, E2)] = (v + w[j], (P, E), j)
        f = g
    # NOTE: parent pointers refer to the state dictionary of the previous stage; we rebuild the set
    best = max(f.items(), key=lambda kv: kv[1][0])
    return best[1][0]
