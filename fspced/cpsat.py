"""Constraint programming model of FSP-CED solved with OR-Tools CP-SAT."""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from ortools.sat.python import cp_model

from .instance import Instance


def solve_cp(inst: Instance, time_limit: float = 60.0, workers: int = 1, hint: Optional[Dict] = None,
             seed: int = 0, log: bool = False, model: str = "credit", ub: Optional[float] = None,
             strengthen: bool = False) -> dict:
    """model = 'pair'  : pairwise crediting booleans z_jk (C_j < C_k), O(n * #givers) booleans;
       model = 'credit': time-indexed credit profile E_t and last completion time Cmax,
                         O(#givers * (Dbar - D0)) booleans (default, see Proposition 2)."""
    n, m, D0, Db = inst.n, inst.m, inst.D0, inst.Dbar
    mdl = cp_model.CpModel()
    present, start, end = [], [], []
    mode_vars = []
    per_battery = [[] for _ in range(m)]
    for j, t in enumerate(inst.targets):
        pj = mdl.NewBoolVar(f"y{j}")
        pmin = min(md.p for md in t.modes)
        if pmin > Db:                      # target can never be engaged
            pmin = 0
        S = mdl.NewIntVar(0, max(0, Db - pmin), f"S{j}")
        C = mdl.NewIntVar(pmin, Db, f"C{j}")
        ys = []
        for h, md in enumerate(t.modes):
            if md.p > Db:
                ys.append(None)
                continue
            yh = mdl.NewBoolVar(f"y{j}_{h}")
            iv = mdl.NewOptionalIntervalVar(S, md.p, C, yh, f"I{j}_{h}")
            for i in md.batteries:
                per_battery[i].append(iv)
            ys.append(yh)
        mdl.Add(sum(y for y in ys if y is not None) == pj)
        present.append(pj)
        start.append(S)
        end.append(C)
        mode_vars.append(ys)
    for i in range(m):
        if per_battery[i]:
            mdl.AddNoOverlap(per_battery[i])
    givers = [j for j in range(n) if inst.targets[j].delta > 0]
    zvars = {}
    if model == "credit":
        # Cmax >= C_k for engaged k; for every t in (D0, Dbar]: Cmax >= t  =>  t - D0 <= E_t,
        # with E_t = sum_j delta_j c_jt and c_jt -> (y_j and C_j <= t-1).
        Cmax = mdl.NewIntVar(0, Db, "Cmax")
        for k in range(n):
            mdl.Add(Cmax >= end[k]).OnlyEnforceIf(present[k])
        if not givers:
            mdl.Add(Cmax <= D0)
        else:
            mdl.Add(Cmax <= D0 + sum(inst.targets[j].delta for j in givers))
            prev = {}
            for t in range(D0 + 1, Db + 1):
                lt = mdl.NewBoolVar(f"l{t}")
                mdl.Add(Cmax <= t - 1).OnlyEnforceIf(lt.Not())
                terms = []
                for j in givers:
                    if min(md.p for md in inst.targets[j].modes) > t - 1:
                        continue
                    c = mdl.NewBoolVar(f"c{j}_{t}")
                    mdl.AddImplication(c, present[j])
                    mdl.Add(end[j] <= t - 1).OnlyEnforceIf(c)
                    if j in prev:
                        mdl.AddImplication(prev[j], c)   # monotone in t (redundant)
                    prev[j] = c
                    terms.append(inst.targets[j].delta * c)
                if terms:
                    mdl.Add(sum(terms) >= t - D0).OnlyEnforceIf(lt)
                else:
                    mdl.Add(lt == 0)
    for k in range(n if model == "pair" else 0):
        terms = []
        for j in givers:
            if j == k:
                continue
            z = mdl.NewBoolVar(f"z{j}_{k}")
            mdl.AddImplication(z, present[j])
            mdl.Add(end[j] + 1 <= end[k]).OnlyEnforceIf(z)
            zvars[(j, k)] = z
            terms.append(inst.targets[j].delta * z)
        if terms:
            mdl.Add(end[k] <= D0 + sum(terms)).OnlyEnforceIf(present[k])
        else:
            mdl.Add(end[k] <= D0).OnlyEnforceIf(present[k])
    # a pair can credit each other in at most one direction
    for (j, k), z in zvars.items():
        if j < k and (k, j) in zvars:
            mdl.AddBoolOr([z.Not(), zvars[(k, j)].Not()])
    # redundant energy cut: total battery-time used <= m * (deadline cap)
    mdl.Add(sum(md.area * y for t, ys in zip(inst.targets, mode_vars)
                for md, y in zip(t.modes, ys) if y is not None) <= m * Db)
    if strengthen:         # same redundant constraints as the flow-shop model: caps, load bounds and ladders
        from .lbbd import master_rows
        R = master_rows(inst)
        var = {}
        for (j, h), q in R["xidx"].items():
            if mode_vars[j][h] is not None:
                var[q] = mode_vars[j][h]
        for j in range(n):
            var[R["yidx"][j]] = present[j]
        for c, lo, up in zip(R["rows"][n:], R["lo"][n:], R["up"][n:]):     # the first n rows link x and y
            if up < 1e18:
                mdl.Add(sum(int(v) * var[q] for q, v in c.items()) <= int(up))
    if ub is not None:     # externally certified upper bound (e.g. from the LBBD master)
        mdl.Add(sum(inst.targets[j].w * present[j] for j in range(n)) <= int(ub))
    mdl.Maximize(sum(inst.targets[j].w * present[j] for j in range(n)))
    if hint:
        for j in range(n):
            if j in hint:
                h, S0 = hint[j]
                mdl.AddHint(present[j], 1)
                mdl.AddHint(start[j], S0)
                for hh, y in enumerate(mode_vars[j]):
                    if y is not None:
                        mdl.AddHint(y, 1 if hh == h else 0)
            else:
                mdl.AddHint(present[j], 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log
    t0 = time.time()
    status = solver.Solve(mdl)
    wall = time.time() - t0
    sol = {}
    val = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        val = int(round(solver.ObjectiveValue()))
        for j in range(n):
            if solver.Value(present[j]):
                h = next(hh for hh, y in enumerate(mode_vars[j]) if y is not None and solver.Value(y))
                sol[j] = (h, solver.Value(start[j]))
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        bound = float(solver.BestObjectiveBound())
    else:   # no solution: CP-SAT does not report a meaningful bound, fall back to a trivial valid one
        bound = float(ub) if ub is not None else float(sum(t.w for t in inst.targets))
    return {"status": solver.StatusName(status), "value": val,
            "bound": bound, "time": wall, "sol": sol,
            "optimal": status == cp_model.OPTIMAL}
