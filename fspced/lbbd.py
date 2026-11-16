"""Logic-based Benders decomposition for the FSP-CED (version 3: flow-shop subproblem, Johnson inequalities
and deficit cuts).

Master problem (0-1 program over the targets and their modes; Gurobi, HiGHS or CP-SAT).  It contains
  * one mode per engaged target;
  * for every family F of pairwise conflicting modes (the modes of a battery, or the modes that use at least two
    batteries of a triple): the load of F is at most the cap, and the JOHNSON INEQUALITIES of F.  The engagements
    of F are sequential, so, with the extensions earned outside F granted from the start, they must pass the
    feasibility test of the single-battery case, which is Johnson's rule.  In Johnson's order of F, for every
    position k:   sum_{l <= k} p_l x_l - sum_{l < k} delta_l x_l - (extensions outside F) <= D0.
    The inequality of the last position is the load bound of the earlier versions of the code;
  * the aggregate Johnson inequalities, in which the m batteries are pooled (times a_l / m);
  * ladder inequalities: an engagement longer than D0 needs the extensions of strictly shorter engagements;
  * Benders cuts.
Subproblem (CP-SAT, flow-shop model of fspced/flowshop.py).  The selection of the master is checked first with
the modes of the master and then with free modes.  An infeasible selection gives a core C, and the cut is the
DEFICIT INEQUALITY

    sum_{j in C} pi_j (1 - [j engaged in an allowed mode]) + sum_{g extending, outside} delta_g [g engaged] >= theta,

where theta is (a lower bound on) the smallest increase of D0 and of the cap that makes C schedulable, pi_j is the
shortest allowed duration of j, and coefficients are truncated at theta.  With theta = 1 this is the lifted
no-good cut of the earlier versions (cuts='nogood'); cuts='standard' drops the second sum, which is NOT valid with
an expandable deadline and is kept only to measure the effect of the flaw.  If the core is infeasible on the
batteries alone (cap-only relaxation), the second sum is dropped validly (overrun cut).
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional

import numpy as np
from ortools.sat.python import cp_model

from .instance import Instance
from . import flowshop as fs

INF = float("inf")


def givers_of(inst: Instance) -> List[int]:
    return [j for j, t in enumerate(inst.targets) if t.delta > 0]


# ----------------------------------------------------------------------------- conflict families
def triangle_families(inst: Instance):
    """Families of pairwise conflicting modes without a common battery: for three batteries a, b, c, the modes
    that use at least two of them.  Returns lists of (j, h), for the triples whose family is not contained in
    the family of a single battery."""
    out = []
    modes = [(j, h, set(md.batteries)) for j, t in enumerate(inst.targets) for h, md in enumerate(t.modes)
             if md.p <= inst.Dbar]
    for tri in itertools.combinations(range(inst.m), 3):
        T = set(tri)
        fam = [(j, h) for j, h, B in modes if len(B & T) >= 2]
        if not fam:
            continue
        common = set(tri)
        for j, h, B in modes:
            if len(B & T) >= 2:
                common &= B
        if not common:
            out.append(fam)
    return out


def conflict_families(inst: Instance):
    fams = []
    for i in range(inst.m):
        fam = [(j, h) for j, t in enumerate(inst.targets) for h, md in enumerate(t.modes)
               if md.p <= inst.Dbar and i in md.batteries]
        if fam:
            fams.append(fam)
    return fams + triangle_families(inst)


# ----------------------------------------------------------------------------- master rows
def master_rows(inst: Instance, ladder: bool = True):
    """Rows of the initial master problem.  Variables: x_(j,h) (binary) and y_j (binary).  For every family F of
    pairwise conflicting modes: load(F) <= Dbar and load(F) <= D0 + total extension (the Johnson inequality of the
    last position); ladder=True adds the ladder inequalities, otherwise the weaker condition of the earlier
    versions ('a target ends by D0 plus the extensions of the others').  The other Johnson inequalities are
    separated in closed form (johnson_cuts)."""
    n, m, D0, Db = inst.n, inst.m, inst.D0, inst.Dbar
    xidx = {}
    for j, t in enumerate(inst.targets):
        for h, md in enumerate(t.modes):
            if md.p <= Db:
                xidx[(j, h)] = len(xidx)
    nx = len(xidx)
    yidx = {j: nx + j for j in range(n)}
    nv = nx + n
    rows, lo, up = [], [], []

    def row(c, l, u):
        rows.append(c); lo.append(l); up.append(u)

    P = {jh: inst.targets[jh[0]].modes[jh[1]].p for jh in xidx}
    dl = [t.delta for t in inst.targets]
    G = [j for j in range(n) if dl[j] > 0]
    for j in range(n):
        c = {q: 1.0 for (jj, h), q in xidx.items() if jj == j}
        c[yidx[j]] = -1.0
        row(c, 0.0, 0.0)
    fams = [[jh for jh in fam if jh in xidx] for fam in conflict_families(inst)]
    for fam in fams:
        row({xidx[jh]: float(P[jh]) for jh in fam}, -INF, float(Db))          # load <= cap
        if G:
            c = {xidx[jh]: float(P[jh]) for jh in fam}
            for g in G:
                c[yidx[g]] = -float(dl[g])
            row(c, -INF, float(D0))                                           # load <= D0 + extensions
    for (j, h), q in xidx.items():
        if P[(j, h)] > D0:
            c = {q: float(P[(j, h)])}
            if ladder:                    # only strictly shorter engagements can have ended before
                for (g, hh), q2 in xidx.items():
                    if g != j and dl[g] > 0 and P[(g, hh)] < P[(j, h)]:
                        c[q2] = -float(dl[g])
            else:
                for k in G:
                    if k != j:
                        c[yidx[k]] = -float(dl[k])
            row(c, -INF, float(D0))
    w = np.zeros(nv)
    for j in range(n):
        w[yidx[j]] = inst.targets[j].w
    return {"nv": nv, "nx": nx, "rows": rows, "lo": lo, "up": up, "w": w, "xidx": xidx, "yidx": yidx,
            "families": fams, "P": P}


def _johnson_families(inst: Instance, R: dict):
    """Families on which the Johnson test is applied: (key, ordered engagements, times, scale, set).  The last
    one pools the m batteries (times a / m, written times m)."""
    xidx, P = R["xidx"], R["P"]
    dl = [t.delta for t in inst.targets]
    out = []
    for q, fam in enumerate(R["families"]):
        order = sorted(fam, key=lambda jh: (fs.johnson_key(P[jh], dl[jh[0]]), jh))
        out.append((q, order, P, 1, set(fam)))
    if inst.m > 1:
        m = inst.m
        A = {jh: inst.targets[jh[0]].modes[jh[1]].area for jh in xidx}
        order = sorted(xidx, key=lambda jh: ((0, A[jh]) if A[jh] <= m * dl[jh[0]] else (1, -dl[jh[0]]), jh))
        out.append(("pool", order, A, m, set(xidx)))
    return out


def johnson_violated(inst: Instance, R: dict, S: List[int], modes: Dict[int, int]):
    """Johnson test of a selection, in closed form.  For every conflict family F, the selected engagements of F
    are scanned in Johnson's order with times (p, delta); the prefix condition of the single-battery case must
    hold when the extensions earned outside F are granted from the start:
        sum_{l <= k in F} p_l x_l - sum_{l < k in F} delta_l x_l - sum_{l not in F} delta_l x_l <= D0.
    Returns the keys of the families with a violated inequality."""
    D0 = inst.D0
    dl = [t.delta for t in inst.targets]
    if not any(dl):
        return []
    if "jfam" not in R:
        R["jfam"] = _johnson_families(inst, R)
    sel = {(j, modes[j]) for j in S}
    bad = []
    for key, order, time_of, scale, fset in R["jfam"]:
        outside = sum(dl[j] for (j, h) in sel if (j, h) not in fset)
        pref = 0
        for jh in order:
            if jh not in sel:
                continue
            if pref + time_of[jh] - scale * outside > scale * D0:
                bad.append(key)
                break
            pref += time_of[jh] - scale * dl[jh[0]]
    return bad


def johnson_rows(inst: Instance, R: dict, keys=None):
    """Johnson inequalities of the families in `keys` (all families if None), one per extending position, as
    cuts (coefs, rhs) in '>=' form."""
    xidx = R["xidx"]
    D0 = inst.D0
    dl = [t.delta for t in inst.targets]
    cuts = []
    if not any(dl):
        return cuts
    if "jfam" not in R:
        R["jfam"] = _johnson_families(inst, R)
    for key, order, time_of, scale, fset in R["jfam"]:
        if keys is not None and key not in keys:
            continue
        out_c = {q: float(scale * dl[jh[0]]) for jh, q in xidx.items() if dl[jh[0]] > 0 and jh not in fset}
        pref = {}
        for jh in order:
            if dl[jh[0]] > 0:
                c = dict(out_c)
                c.update(pref)
                c[xidx[jh]] = -float(time_of[jh])
                cuts.append((c, -float(scale * D0)))
            coef = time_of[jh] - scale * dl[jh[0]]
            if coef:
                pref[xidx[jh]] = -float(coef)
    return cuts


class Master:
    """Master problem kept as a list of rows; solved by Gurobi (persistent model, cuts added incrementally), by
    HiGHS through scipy, or by CP-SAT.  A cut is a pair (coefficients, rhs) meaning  sum coef * var >= rhs."""

    def __init__(self, inst: Instance, backend: str = "highs", threads: int = 2, ladder: bool = True):
        self.inst, self.backend, self.cuts, self.ncut, self.threads = inst, backend, [], 0, threads
        self.R = R = master_rows(inst, ladder)
        nv = R["nv"]
        if backend == "gurobi":
            import gurobipy as gp
            from gurobipy import GRB
            md_ = gp.Model()
            md_.setParam("OutputFlag", 0); md_.setParam("Threads", threads); md_.setParam("MIPGap", 1e-9)
            self.x = md_.addVars(nv, obj=list(R["w"]), vtype=GRB.BINARY)
            md_.ModelSense = GRB.MAXIMIZE
            for c, l, u in zip(R["rows"], R["lo"], R["up"]):
                e = gp.LinExpr(list(c.values()), [self.x[q] for q in c])
                if l == u:
                    md_.addLConstr(e, GRB.EQUAL, l)
                else:
                    if l > -INF:
                        md_.addLConstr(e, GRB.GREATER_EQUAL, l)
                    if u < INF:
                        md_.addLConstr(e, GRB.LESS_EQUAL, u)
            self.model, self.lbc = md_, None
        elif backend == "highs":
            from scipy.sparse import coo_matrix
            ri, ci, vv = [], [], []
            for r, c in enumerate(R["rows"]):
                for q, v in c.items():
                    ri.append(r); ci.append(q); vv.append(v)
            self.A0 = coo_matrix((vv, (ri, ci)), shape=(len(R["rows"]), nv)).tocsr()

    def solve(self, time_limit: float, lb: int = 0):
        """Returns (status, selection, modes, bound); status in {'optimal', 'feasible', 'infeasible', 'unknown'}."""
        R, n = self.R, self.inst.n
        if self.backend == "gurobi":
            import gurobipy as gp
            from gurobipy import GRB
            md_ = self.model
            while self.ncut < len(self.cuts):
                c, rhs = self.cuts[self.ncut]
                md_.addLConstr(gp.LinExpr(list(c.values()), [self.x[q] for q in c]), GRB.GREATER_EQUAL, rhs)
                self.ncut += 1
            if self.lbc is not None:
                md_.remove(self.lbc)
            self.lbc = md_.addConstr(gp.quicksum(R["w"][q] * self.x[q] for q in R["yidx"].values()) >= lb + 1) \
                if lb > 0 else None
            md_.setParam("TimeLimit", float(max(1.0, time_limit)))
            md_.optimize()
            if md_.Status == GRB.INFEASIBLE:
                return "infeasible", None, None, lb
            if md_.SolCount == 0:
                return "unknown", None, None, float(md_.ObjBound)
            xv = [self.x[q].X for q in range(R["nv"])]
            st = "optimal" if md_.Status == GRB.OPTIMAL else "feasible"
            bound = float(md_.ObjBound)
        elif self.backend == "highs":
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import coo_matrix, vstack
            extra, elo, eup = [], [], []
            for c, rhs in self.cuts:
                extra.append(c); elo.append(rhs); eup.append(INF)
            if lb > 0:
                extra.append({q: R["w"][q] for q in R["yidx"].values()}); elo.append(lb + 1.0); eup.append(INF)
            A = self.A0
            if extra:
                ri, ci, vv = [], [], []
                for r, c in enumerate(extra):
                    for q, v in c.items():
                        ri.append(r); ci.append(q); vv.append(v)
                A = vstack([A, coo_matrix((vv, (ri, ci)), shape=(len(extra), R["nv"])).tocsr()]).tocsr()
            res = milp(-R["w"], constraints=LinearConstraint(A, np.array(R["lo"] + elo), np.array(R["up"] + eup)),
                       integrality=np.ones(R["nv"]), bounds=Bounds(0, 1),
                       options={"time_limit": float(max(1.0, time_limit)), "mip_rel_gap": 1e-9})
            if res.status == 2:
                return "infeasible", None, None, lb
            if res.x is None:
                db = getattr(res, "mip_dual_bound", None)      # None when HiGHS stops before any bound is known
                return "unknown", None, None, (-float(db) if db is not None else INF)
            xv = res.x
            st = "optimal" if res.status == 0 else "feasible"
            bound = -float(res.mip_dual_bound) if getattr(res, "mip_dual_bound", None) is not None else -res.fun
        else:                                         # CP-SAT (no MILP solver needed)
            mdl = cp_model.CpModel()
            v = [mdl.NewBoolVar(f"b{q}") for q in range(R["nv"])]
            for c, l, u in zip(R["rows"], R["lo"], R["up"]):
                e = sum(int(val) * v[q] for q, val in c.items())
                if l == u:
                    mdl.Add(e == int(l))
                else:
                    if l > -INF:
                        mdl.Add(e >= int(l))
                    if u < INF:
                        mdl.Add(e <= int(u))
            for c, rhs in self.cuts:
                mdl.Add(sum(int(val) * v[q] for q, val in c.items()) >= int(rhs))
            obj = sum(int(R["w"][q]) * v[q] for q in R["yidx"].values())
            if lb > 0:
                mdl.Add(obj >= lb + 1)
            mdl.Maximize(obj)
            s = cp_model.CpSolver()
            s.parameters.max_time_in_seconds = max(1.0, time_limit)
            s.parameters.num_search_workers = self.threads
            stc = s.Solve(mdl)
            if stc == cp_model.INFEASIBLE:
                return "infeasible", None, None, lb
            if stc not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return "unknown", None, None, float(s.BestObjectiveBound())
            xv = [s.Value(x) for x in v]
            st = "optimal" if stc == cp_model.OPTIMAL else "feasible"
            bound = float(s.BestObjectiveBound())
        S = [j for j in range(n) if xv[R["yidx"][j]] > 0.5]
        modes = {j: h for (j, h), q in R["xidx"].items() if xv[q] > 0.5}
        return st, S, modes, bound


# ----------------------------------------------------------------------------- pairwise subproblem (ablation)
def _pair_model(inst: Instance, U: List[int], allowed=None, mode_hint=None):
    """Subproblem of the earlier versions: crediting literals z_jk for every extending j and every k."""
    m, D0, Db = inst.m, inst.D0, inst.Dbar
    mdl = cp_model.CpModel()
    pres, start, end, ys = {}, {}, {}, {}
    per_b = [[] for _ in range(m)]
    for j in U:
        t = inst.targets[j]
        hs = [h for h in (allowed[j] if allowed and j in allowed else range(len(t.modes))) if t.modes[h].p <= Db]
        pj = mdl.NewBoolVar(f"y{j}")
        pmin = min([t.modes[h].p for h in hs], default=0)
        Sv = mdl.NewIntVar(0, max(0, Db - pmin), f"S{j}")
        Cv = mdl.NewIntVar(pmin, Db, f"C{j}")
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
        pres[j], start[j], end[j], ys[j] = pj, Sv, Cv, lits
    for i in range(m):
        if len(per_b[i]) > 1:
            mdl.AddNoOverlap(per_b[i])
    G = [j for j in U if inst.targets[j].delta > 0]
    z = {}
    for k in U:
        terms = []
        for j in G:
            if j == k:
                continue
            zz = mdl.NewBoolVar(f"z{j}_{k}")
            mdl.AddImplication(zz, pres[j])
            mdl.Add(end[j] + 1 <= end[k]).OnlyEnforceIf(zz)
            z[(j, k)] = zz
            terms.append(inst.targets[j].delta * zz)
        mdl.Add(end[k] <= D0 + sum(terms)).OnlyEnforceIf(pres[k])
    for (j, k), zz in z.items():
        if j < k and (k, j) in z:
            mdl.AddBoolOr([zz.Not(), z[(k, j)].Not()])
    return mdl, pres, start, ys


def check_pair(inst: Instance, S: List[int], time_limit: float, modes=None, fixed: bool = False,
               extra: Optional[List[int]] = None, workers: int = 1):
    """Same interface as flowshop.check, with the pairwise model."""
    S = list(S)
    extra = [j for j in (extra or []) if j not in set(S)]
    allowed = {j: [modes[j]] for j in S + extra if j in modes} if fixed else None
    mdl, pres, start, ys = _pair_model(inst, S + extra, allowed, None if fixed else modes)
    lits = [pres[j] for j in S]
    mdl.AddAssumptions(lits)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, time_limit)
    solver.parameters.num_search_workers = workers
    st = solver.Solve(mdl)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = {j: (next(h for h, l in ys[j] if solver.Value(l)), solver.Value(start[j]))
               for j in S + extra if solver.Value(pres[j])}
        return "feasible", sol
    if st == cp_model.INFEASIBLE:
        idx = set(solver.SufficientAssumptionsForInfeasibility())
        C = [j for l, j in zip(lits, S) if l.Index() in idx]
        return "infeasible", (C if C else S)
    return "unknown", None


def repair(inst: Instance, S: List[int], time_limit: float, hint=None, workers: int = 1):
    """Best plan that engages a subset of S (flow-shop model, short limit); updates the incumbent."""
    fm = fs.FlowModel(inst, S)
    fm.mdl.Maximize(sum(inst.targets[j].w * fm.pres[j] for j in S))
    if hint:
        for j in S:
            fm.mdl.AddHint(fm.pres[j], 1 if j in hint else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, time_limit)
    solver.parameters.num_search_workers = workers
    st = solver.Solve(fm.mdl)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return fm.plan(solver)
    return {}


# ----------------------------------------------------------------------------- cuts
def shrink(checker, inst: Instance, C: List[int], modes, fixed: bool, helpers: List[int], budget: float,
           per_check: float = 1.0):
    """Deletion filter: drop targets from the core while it stays infeasible.  A target removed from the core
    becomes a helper (it stays optional in the test), so that the final pair (core, helpers) is a certificate."""
    t0 = time.time()
    C, helpers = list(C), list(helpers)
    for j in sorted(C, key=lambda j: inst.min_area(j)):       # small targets are the least likely culprits
        if time.time() - t0 > budget or len(C) <= 1:
            break
        if j not in C:
            continue
        rest = [k for k in C if k != j]
        hp = helpers + [k for k in C if k not in rest]
        st, res = checker(inst, rest, per_check, modes=modes, fixed=fixed, extra=hp)
        if st == "infeasible":
            new = [k for k in rest if k in set(res)] if res else rest
            helpers = hp + [k for k in rest if k not in set(new)]
            C = new
    return C, helpers


def deficit_cut(inst: Instance, R: dict, C: List[int], helpers: List[int], modes, fixed: bool, theta: int,
                outside: bool = True):
    """Deficit inequality of a certificate.  C: core; helpers: targets that were optional in the test; a target
    of C or of the helpers is restricted to its mode in `modes` if fixed, else it may use any mode.
        sum_{j in C} min(pi_j, theta) (1 - [j in an allowed mode])
          + sum_{g extending, not in C} min(delta_g, theta) ([g engaged] - [g helper in an allowed mode]) >= theta.
    outside=False drops the second sum (valid for an overrun certificate; NOT valid for cuts='standard')."""
    xidx, yidx = R["xidx"], R["yidx"]
    theta = max(1, int(theta))
    c, rhs = {}, float(theta)
    Cset, Hset = set(C), set(helpers)

    def add(q, v):
        c[q] = c.get(q, 0.0) + v

    for j in C:
        hs = [modes[j]] if fixed else [h for h in range(len(inst.targets[j].modes)) if (j, h) in xidx]
        pi = min(theta, min(inst.targets[j].modes[h].p for h in hs))
        rhs -= pi                                  # pi * (1 - sum x)  ->  -pi * sum x >= theta - pi
        if fixed:
            add(xidx[(j, modes[j])], -float(pi))
        else:
            add(yidx[j], -float(pi))
    if outside:
        for g in givers_of(inst):
            if g in Cset:
                continue
            d = float(min(theta, inst.targets[g].delta))
            if g in Hset:
                if fixed:                          # engaged in a mode other than the allowed one
                    add(yidx[g], d)
                    add(xidx[(g, modes[g])], -d)
            else:
                add(yidx[g], d)
    return c, rhs


# ----------------------------------------------------------------------------- sufficient tests
def _decoder_check(inst: Instance, S: List[int], modes: Dict[int, int]):
    """Fast sufficient test: SGS-ED engages every target of S (orders: credit first / density)."""
    from .heur import Decoder
    dec = Decoder(inst, passes=3)
    A = dec.A
    Sset = set(S)
    dens = {j: A.w[j] / A.minarea[j] for j in S}
    first = sorted([j for j in S if A.delta[j] > 0 and A.delta[j] >= A.minp[j]], key=lambda j: A.minp[j])
    mid = sorted([j for j in S if A.delta[j] > 0 and A.delta[j] < A.minp[j]], key=lambda j: -A.delta[j])
    rest = sorted([j for j in S if A.delta[j] == 0], key=lambda j: -A.minarea[j])
    outs = [j for j in range(inst.n) if j not in Sset]
    orders = [first + mid + rest, first + mid + sorted(rest, key=lambda j: -dens[j]),
              sorted(S, key=lambda j: -A.minarea[j])]
    for o in orders:
        for genes in ("master", -1, -2):
            rules = np.full(inst.n, -1, dtype=np.int64)
            for j in S:
                rules[j] = modes.get(j, -1) if genes == "master" else genes
            order = np.array(o + outs, dtype=np.int64)
            v, _, mode, start = dec(order, rules)
            sol = {int(j): (int(mode[j]), int(start[j])) for j in S if mode[j] >= 0}
            if len(sol) == len(S):
                return sol
    return None


# ----------------------------------------------------------------------------- driver
def solve_lbbd(inst: Instance, time_limit: float = 300.0, ig_time: float = 5.0, workers: int = 2,
               sub_limit: float = 10.0, shrink_max: int = 40, seed: int = 0, log: bool = False,
               master: str = "auto", fallback: bool = True, cuts: str = "deficit", sub: str = "flow",
               johnson=True, max_unknown: int = 3, relax_limit: float = 1.0, free_frac: float = 0.3,
               fix_modes: bool = True) -> dict:
    """LBBD driver.
    cuts : 'deficit' (default), 'nogood' (deficit inequality with theta = 1, i.e. the lifted no-good cut) or
           'standard' (classical no-good cut, NOT valid with an expandable deadline; ablation only).
    sub  : 'flow' (flow-shop subproblem) or 'pair' (pairwise crediting literals; ablation).
    johnson : False switches off the Johnson and ladder inequalities (ablation).
    fix_modes : False skips the test of the selection with the modes of the master problem, so that the subproblem
           only sees the targets and chooses the modes itself (decomposition in the space of targets; ablation).
    The bound is certified as long as every cut comes from a decided subproblem; if a selection has to be cut
    without proof, the last certified bound is kept and the search hands over to the monolithic model."""
    assert cuts in ("deficit", "nogood", "standard") and sub in ("flow", "pair")
    from .heur import ig
    from .check import check as verify
    t0 = time.time()
    left = lambda: time_limit - (time.time() - t0)
    g = ig(inst, ig_time, seed=seed) if ig_time > 0 else {"value": 0, "sol": {}}
    best_val, best_sol = g["value"], dict(g["sol"])
    if master == "auto":
        try:
            from .milp import backend_available
            master = "gurobi" if backend_available("gurobi") else "highs"
        except Exception:
            master = "highs"
    M = Master(inst, backend=master, threads=workers, ladder=bool(johnson))
    jdone = set()                                  # families whose Johnson inequalities are in the master
    if johnson == "full":
        M.cuts.extend(johnson_rows(inst, M.R))
        jdone = None
    checker = fs.check if sub == "flow" else check_pair
    G = givers_of(inst)
    UB = float(sum(t.w for t in inst.targets))
    certified = True
    it = n_unknown = n_dec = n_fixed = n_free = n_over = n_def = n_john = n_sub = n_jfail = 0
    theta_sum = 0
    t_master = t_sub = 0.0
    status = "time_limit"
    while left() > 0:
        it += 1
        tm = time.time()
        st, S, modes, bound = M.solve(left(), lb=best_val)
        t_master += time.time() - tm
        if st == "infeasible":                    # no selection is better than the incumbent
            if certified:
                UB, status = best_val, "optimal"
            break
        if certified:
            UB = min(UB, max(float(np.floor(bound + 1e-6)), best_val))
        if st == "unknown":
            break
        val = sum(inst.targets[j].w for j in S)
        # ---- level 1: Johnson test in closed form; the inequalities of a family enter the master the first
        #      time the family fails the test
        if johnson and jdone is not None:
            bad = [k for k in johnson_violated(inst, M.R, S, modes) if k not in jdone]
            if bad:
                jc = johnson_rows(inst, M.R, set(bad))
                M.cuts.extend(jc)
                jdone.update(bad)
                n_john += len(jc)
                n_jfail += 1
                if log:
                    print(f"it {it}: master {val} (UB {UB:.0f}), Johnson test failed on {len(bad)} families, "
                          f"{len(jc)} inequalities added, t={time.time() - t0:.1f}", flush=True)
                continue
        # ---- level 2: decoder, then the flow shop with the modes of the master, then with free modes
        ts = time.time()
        n_sub += 1
        sol = _decoder_check(inst, S, modes) if sub_limit > 0 else None
        cert = None                                # (core, helpers, fixed)
        if sol is not None:
            n_dec += 1
        elif sub_limit > 0:                        # sub_limit <= 0: every subproblem is left undecided (tests only)
            s1, r1 = ("unknown", None)
            if fix_modes:
                s1, r1 = checker(inst, S, min(sub_limit, max(0.5, left())), modes=modes, fixed=True,
                                 workers=workers)
            if s1 == "feasible":
                sol = r1
                n_fixed += 1
            else:
                # a selection proven infeasible with its modes gets the full limit with free modes, because the
                # core over the targets gives a stronger cut; an undecided one gets a fraction of it
                frac = free_frac if (fix_modes and s1 != "infeasible") else 1.0
                s2, r2 = checker(inst, S, min(frac * sub_limit, max(0.5, left())), modes=modes, fixed=False,
                                 workers=workers)
                if s2 == "feasible":
                    sol = r2
                    n_free += 1
                elif s2 == "infeasible":
                    cert = (r2, [j for j in S if j not in set(r2)], False)
                elif s1 == "infeasible":
                    cert = (r1, [j for j in S if j not in set(r1)], True)
        if sol is not None:
            t_sub += time.time() - ts
            ok, v, _ = verify(inst, sol)
            assert ok and v >= val, (ok, v, val)
            if v > best_val:
                best_val, best_sol = v, sol
            if best_val >= UB - 1e-9:
                status = "optimal" if certified else status
                break
            continue
        if cert is None:                           # undecided: the selection is cut without proof
            n_unknown += 1
            certified = False
            if fallback and (n_unknown >= max_unknown or time.time() - t0 > 0.4 * time_limit):
                status = "fallback"
                t_sub += time.time() - ts
                break
            M.cuts.append(deficit_cut(inst, M.R, list(S), [], modes, fix_modes, 1))
        else:
            C, helpers, fixed = cert
            if len(C) <= shrink_max and left() > 1:
                C, helpers = shrink(checker, inst, C, modes, fixed, helpers,
                                    budget=min(5.0, 0.05 * time_limit))
            if cuts == "standard":
                M.cuts.append(deficit_cut(inst, M.R, C, helpers, modes, fixed, 1, outside=False))
            elif cuts == "nogood":
                M.cuts.append(deficit_cut(inst, M.R, C, helpers, modes, fixed, 1))
            else:
                allowed = {j: [modes[j]] for j in C + helpers} if fixed else None
                over = fs.relaxation(inst, C, min(relax_limit, max(0.2, left())), allowed=allowed,
                                     cap_only=True, workers=workers)
                if not G:                          # common deadline: an infeasible core overruns the deadline
                    over = max(1, over)
                if over >= 1:                      # infeasible on the batteries alone: no outside terms
                    M.cuts.append(deficit_cut(inst, M.R, C, helpers, modes, fixed, over, outside=False))
                    n_over += 1
                    theta_sum += over
                if G:
                    th = max(1, fs.relaxation(inst, C, min(relax_limit, max(0.2, left())), allowed=allowed,
                                              helpers=helpers, workers=workers))
                    if over < 1 or th > over:
                        M.cuts.append(deficit_cut(inst, M.R, C, helpers, modes, fixed, th))
                        n_def += 1
                        theta_sum += th
        rep = repair(inst, S, min(2.0, max(0.2, left())), hint=best_sol, workers=workers)
        rv = sum(inst.targets[j].w for j in rep)
        if rv > best_val:
            ok, v, _ = verify(inst, rep)
            if ok:
                best_val, best_sol = rv, rep
        t_sub += time.time() - ts
        if log:
            print(f"it {it}: master {val} (UB {UB:.0f}), cuts {len(M.cuts)}, LB {best_val}, "
                  f"t={time.time() - t0:.1f}", flush=True)
        if best_val >= UB - 1e-9:
            status = "optimal" if certified else status
            break
    n_fb = 0
    if status == "fallback" or (status == "time_limit" and fallback and left() > 1):
        r = fs.solve_cpf(inst, max(1.0, left()), workers=workers, hint=best_sol, seed=seed, ub=UB)
        n_fb = 1
        if r["value"] is not None and r["value"] > best_val:
            best_val, best_sol = r["value"], r["sol"]
        if r["status"] in ("OPTIMAL", "FEASIBLE"):
            UB = min(UB, max(float(np.floor(r["bound"] + 1e-6)), best_val))
    opt = status == "optimal" or best_val >= UB - 1e-9
    return {"value": int(best_val), "bound": float(max(UB, best_val)), "optimal": bool(opt),
            "time": time.time() - t0, "iters": it, "cuts": len(M.cuts), "unknown": n_unknown,
            "subproblems": n_sub, "decoder_hits": n_dec, "fixed_hits": n_fixed, "free_hits": n_free,
            "johnson_cuts": n_john, "johnson_fails": n_jfail, "overrun_cuts": n_over, "deficit_cuts": n_def, "theta_sum": theta_sum,
            "t_master": t_master, "t_sub": t_sub, "fallback": n_fb,
            "status": "OPTIMAL" if opt else "FEASIBLE", "sol": best_sol, "ig_value": g["value"], "master": master,
            "cuts_type": cuts, "sub": sub, "johnson": bool(johnson), "fix_modes": bool(fix_modes),
            "valid": cuts != "standard" or not G}
