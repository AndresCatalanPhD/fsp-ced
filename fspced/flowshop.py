"""Flow-shop view of the expandable deadline (Section 3 of the paper).

A set of engagements meets the expandable deadline if and only if a two-stage flow shop meets a makespan
bound: the first stage is the battery schedule and the second stage is a single machine, common to all
targets, that processes the extension delta_j of every extending target j after its engagement.  Only the
extending targets need a second-stage operation; the other targets only have to end by D0 plus the total
extension earned.  This module turns the equivalence into

  * a CP model with ONE additional disjunctive resource and no crediting literals (`build`), used both as a
    monolithic model (`solve_cpf`) and as the subproblem of the decomposition (`check`);
  * the relaxation value of a set of engagements (`relaxation`): the smallest theta such that the set can be
    engaged when the initial deadline and the cap are both increased by theta (deficit), or when only the cap
    is increased and the expandable deadline is ignored (overrun).
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

from ortools.sat.python import cp_model

from .instance import Instance


def johnson_key(p: int, d: int):
    """Sort key of Johnson's rule for the two-machine flow shop with times (p, d)."""
    return (0, p) if p <= d else (1, -d)


class FlowModel:
    """CP model over a universe of targets.  allowed[j] lists the modes of target j that may be used
    (default: all).  If `theta` is True, an integer variable theta >= 0 relaxes the cap (stage-1 ends by
    Dbar + theta) and, unless `cap_only`, the initial deadline (D0 + theta).  With `cap_only`, the expandable
    deadline is dropped altogether, which leaves the multiprocessor scheduling problem on the batteries."""

    def __init__(self, inst: Instance, universe: Iterable[int], allowed: Optional[Dict[int, List[int]]] = None,
                 theta: bool = False, cap_only: bool = False, theta_max: Optional[int] = None,
                 mode_hint: Optional[Dict[int, int]] = None):
        self.inst = inst
        D0, Db, m = inst.D0, inst.Dbar, inst.m
        mdl = cp_model.CpModel()
        self.U = list(universe)
        self.theta = None
        hor = Db
        if theta:
            if theta_max is None:
                theta_max = sum(max(md.p for md in inst.targets[j].modes) for j in self.U)
            self.theta = mdl.NewIntVar(0, theta_max, "theta")
            hor = Db + theta_max
        self.pres, self.start, self.end, self.ys = {}, {}, {}, {}
        per_b = [[] for _ in range(m)]
        for j in self.U:
            t = inst.targets[j]
            hs = [h for h in (allowed[j] if allowed and j in allowed else range(len(t.modes)))
                  if t.modes[h].p <= hor]
            pj = mdl.NewBoolVar(f"y{j}")
            if not hs:
                mdl.Add(pj == 0)
                self.pres[j], self.start[j], self.end[j], self.ys[j] = pj, mdl.NewConstant(0), mdl.NewConstant(0), []
                continue
            pmin = min(t.modes[h].p for h in hs)
            Sv = mdl.NewIntVar(0, hor - pmin, f"S{j}")
            Cv = mdl.NewIntVar(pmin, hor, f"C{j}")
            lits = []
            for h in hs:
                md = t.modes[h]
                yh = mdl.NewBoolVar(f"y{j}_{h}")
                iv = mdl.NewOptionalIntervalVar(Sv, md.p, Cv, yh, f"I{j}_{h}")
                for i in md.batteries:
                    per_b[i].append(iv)
                lits.append((h, yh))
                if mode_hint and j in mode_hint:
                    mdl.AddHint(yh, 1 if h == mode_hint[j] else 0)
            mdl.Add(sum(l for _, l in lits) == pj)
            if self.theta is not None:
                mdl.Add(Cv <= Db + self.theta).OnlyEnforceIf(pj)
            self.pres[j], self.start[j], self.end[j], self.ys[j] = pj, Sv, Cv, lits
        for i in range(m):
            if len(per_b[i]) > 1:
                mdl.AddNoOverlap(per_b[i])
        # second stage: one operation of length delta_j per extending target, on a common machine
        G = [j for j in self.U if inst.targets[j].delta > 0 and self.ys[j]]
        self.G = G
        if not cap_only:
            tot = sum(inst.targets[j].delta for j in G)
            top = D0 + tot + (theta_max if theta else 0)
            Y = D0 + sum(inst.targets[j].delta * self.pres[j] for j in G)
            if self.theta is not None:
                Y = Y + self.theta
            ivs = []
            for j in G:
                d = inst.targets[j].delta
                s2 = mdl.NewIntVar(0, top - d, f"t{j}")
                ivs.append(mdl.NewOptionalIntervalVar(s2, d, s2 + d, self.pres[j], f"Q{j}"))
                mdl.Add(s2 >= self.end[j]).OnlyEnforceIf(self.pres[j])
                mdl.Add(s2 + d <= Y).OnlyEnforceIf(self.pres[j])
            if len(ivs) > 1:
                mdl.AddNoOverlap(ivs)
            for k in self.U:
                if inst.targets[k].delta == 0 and self.ys[k]:
                    mdl.Add(self.end[k] <= Y).OnlyEnforceIf(self.pres[k])
        self.mdl = mdl

    def plan(self, solver) -> dict:
        sol = {}
        for j in self.U:
            if self.ys[j] and solver.Value(self.pres[j]):
                h = next(h for h, l in self.ys[j] if solver.Value(l))
                sol[j] = (h, solver.Value(self.start[j]))
        return sol


def _solver(time_limit: float, workers: int, seed: int = 0, log: bool = False):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = max(0.05, float(time_limit))
    s.parameters.num_search_workers = workers
    s.parameters.random_seed = seed
    s.parameters.log_search_progress = log
    return s


def check(inst: Instance, S: List[int], time_limit: float, modes: Optional[Dict[int, int]] = None,
          fixed: bool = False, extra: Optional[List[int]] = None, workers: int = 1, core: bool = True):
    """Is there a plan that engages every target of S (and possibly some targets of `extra`)?
    fixed=True: every target of S uses the mode given in `modes`; otherwise the modes are free and `modes` is a
    hint.  Returns (status, plan or core) with status in {'feasible', 'infeasible', 'unknown'}; the core is a
    subset C of S such that no plan engages all of C under the same restrictions."""
    S = list(S)
    Sset = set(S)
    extra = [j for j in (extra or []) if j not in Sset]
    allowed = {j: [modes[j]] for j in S + extra if j in modes} if fixed else None
    fm = FlowModel(inst, S + extra, allowed=allowed, mode_hint=None if fixed else modes)
    lits = [fm.pres[j] for j in S]
    if core:
        fm.mdl.AddAssumptions(lits)
    else:
        for l in lits:
            fm.mdl.Add(l == 1)
    solver = _solver(time_limit, workers)
    st = solver.Solve(fm.mdl)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "feasible", fm.plan(solver)
    if st == cp_model.INFEASIBLE:
        if core:
            idx = set(solver.SufficientAssumptionsForInfeasibility())
            C = [j for l, j in zip(lits, S) if l.Index() in idx]
            return "infeasible", (C if C else S)
        return "infeasible", S
    return "unknown", None


def relaxation(inst: Instance, C: List[int], time_limit: float, allowed: Optional[Dict[int, List[int]]] = None,
               helpers: Optional[List[int]] = None, cap_only: bool = False, workers: int = 1) -> int:
    """Lower bound on the smallest theta such that every target of C can be engaged (in its allowed modes,
    possibly together with some `helpers`) when the cap is raised to Dbar + theta and, unless cap_only, the
    initial deadline to D0 + theta.  The value returned is a proven lower bound (CP-SAT bound), hence valid for
    the cuts even when the time limit stops the search."""
    C = list(C)
    Cset = set(C)
    helpers = [] if cap_only else [j for j in (helpers or []) if j not in Cset]
    hi = sum(min(inst.targets[j].modes[h].p for h in (allowed[j] if allowed and j in allowed
                                                      else range(len(inst.targets[j].modes)))) for j in C)
    fm = FlowModel(inst, C + helpers, allowed=allowed, theta=True, cap_only=cap_only, theta_max=hi)
    for j in C:
        fm.mdl.Add(fm.pres[j] == 1)
    fm.mdl.Minimize(fm.theta)
    solver = _solver(time_limit, workers)
    st = solver.Solve(fm.mdl)
    if st == cp_model.OPTIMAL:
        return int(round(solver.ObjectiveValue()))
    if st == cp_model.FEASIBLE:
        return int(max(0, round(solver.BestObjectiveBound() - 1e-6)))
    return 0


def solve_cpf(inst: Instance, time_limit: float = 60.0, workers: int = 1, hint: Optional[Dict] = None,
              seed: int = 0, log: bool = False, ub: Optional[float] = None, strengthen: bool = True) -> dict:
    """Monolithic flow-shop CP model (CPF).  With strengthen=True, the battery loads and the Johnson
    inequalities of the master problem are added as redundant linear constraints."""
    n, m, D0, Db = inst.n, inst.m, inst.D0, inst.Dbar
    fm = FlowModel(inst, range(n))
    mdl = fm.mdl
    if strengthen:
        from .lbbd import master_rows
        R = master_rows(inst)
        var = {}
        for (j, h), q in R["xidx"].items():
            lit = dict(fm.ys[j]).get(h)
            if lit is not None:
                var[q] = lit
        for j in range(n):
            var[R["yidx"][j]] = fm.pres[j]
        for c, lo, up in zip(R["rows"][n:], R["lo"][n:], R["up"][n:]):     # the first n rows link x and y
            e = sum(int(v) * var[q] for q, v in c.items())
            if up < 1e18:
                mdl.Add(e <= int(up))
    obj = sum(inst.targets[j].w * fm.pres[j] for j in range(n))
    if ub is not None:
        mdl.Add(obj <= int(ub))
    mdl.Maximize(obj)
    if hint:
        for j in range(n):
            if not fm.ys[j]:
                continue
            mdl.AddHint(fm.pres[j], 1 if j in hint else 0)
            if j in hint:
                mdl.AddHint(fm.start[j], hint[j][1])
                for h, l in fm.ys[j]:
                    mdl.AddHint(l, 1 if h == hint[j][0] else 0)
    solver = _solver(time_limit, workers, seed, log)
    t0 = time.time()
    st = solver.Solve(mdl)
    wall = time.time() - t0
    sol, val = {}, None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        val = int(round(solver.ObjectiveValue()))
        sol = fm.plan(solver)
        bound = float(solver.BestObjectiveBound())
    else:
        bound = float(ub) if ub is not None else float(sum(t.w for t in inst.targets))
    return {"status": solver.StatusName(st), "value": val, "bound": bound, "time": wall, "sol": sol,
            "optimal": st == cp_model.OPTIMAL}
