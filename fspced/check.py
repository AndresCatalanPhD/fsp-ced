"""Independent feasibility checker for FSP-CED schedules.

A solution is a dict {target j: (mode index h, start time S_j)} for the engaged targets.
"""
from __future__ import annotations

from typing import Dict, Tuple

from .instance import Instance


def deadline_ok(inst: Instance, comp: Dict[int, int]) -> Tuple[bool, str]:
    """Expandable-deadline condition: every engaged target k must satisfy
    C_k <= min(Dbar, D0 + sum of delta_j over engaged j with C_j < C_k)."""
    for k, Ck in comp.items():
        ext = sum(inst.targets[j].delta for j, Cj in comp.items() if Cj < Ck)
        lim = min(inst.Dbar, inst.D0 + ext)
        if Ck > lim:
            return False, f"target {k} completes at {Ck} > deadline in force {lim}"
    return True, ""


def check(inst: Instance, sol: Dict[int, Tuple[int, int]]) -> Tuple[bool, int, str]:
    """Return (feasible, value, message)."""
    busy = {}
    comp = {}
    for j, (h, S) in sol.items():
        if not (0 <= j < inst.n):
            return False, 0, f"bad target {j}"
        modes = inst.targets[j].modes
        if not (0 <= h < len(modes)):
            return False, 0, f"bad mode {h} for target {j}"
        if S < 0:
            return False, 0, f"negative start for {j}"
        md = modes[h]
        C = S + md.p
        comp[j] = C
        for i in md.batteries:
            for t in range(S, C):
                if (i, t) in busy:
                    return False, 0, f"battery {i} busy at {t} by {busy[(i, t)]} and {j}"
                busy[(i, t)] = j
    ok, msg = deadline_ok(inst, comp)
    if not ok:
        return False, 0, msg
    return True, sum(inst.targets[j].w for j in sol), ""
