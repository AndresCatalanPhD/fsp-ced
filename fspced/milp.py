"""MILP formulations of FSP-CED, solved with Gurobi (gurobipy) when a license is available and
with HiGHS (highspy) otherwise.  Select with backend='gurobi' | 'highs' | 'auto'.

* 'TI'  : time-indexed formulation (x_{jht} = 1 if target j starts at t in mode h) with a
          cumulative credit profile for the expandable deadline.
* 'DJ'  : disjunctive (big-M) formulation in the spirit of the original OPTIMA 2017 model,
          with one start time per target, pairwise order variables and pairwise crediting
          variables.
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

# NOTE: do not import OR-Tools in the same process as highspy (both bundle HiGHS symbols).
_BACKEND = None


def backend_available(name: str) -> bool:
    try:
        if name == "gurobi":
            import gurobipy as gp
            m = gp.Model()
            m.setParam("OutputFlag", 0)
            x = m.addVars(2001, ub=1.0)          # exceeds the size-limited pip license
            m.addConstr(x.sum() <= 1)
            m.optimize()
            return True
        if name == "highs":
            import highspy  # noqa: F401
            return True
    except Exception:
        return False
    return False


def default_backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = "gurobi" if backend_available("gurobi") else "highs"
    return _BACKEND

from .instance import Instance


class _Builder:
    def __init__(self):
        self.lb, self.ub, self.cost, self.integ = [], [], [], []
        self.rows = []      # (lower, upper, idx list, val list)

    def var(self, lb, ub, cost=0.0, integer=False):
        self.lb.append(lb); self.ub.append(ub); self.cost.append(cost); self.integ.append(integer)
        return len(self.lb) - 1

    def row(self, lo, up, idx, val):
        self.rows.append((lo, up, list(idx), list(val)))

    def solve(self, time_limit, relax=False, threads=1, log=False, mip_rel_gap=1e-6, backend="auto"):
        if backend == "auto":
            backend = default_backend()
        if backend == "gurobi":
            r = self._solve_gurobi(time_limit, relax, threads, log, mip_rel_gap)
        else:
            r = self._solve_highs(time_limit, relax, threads, log, mip_rel_gap)
        r["backend"] = backend
        return r

    def _solve_gurobi(self, time_limit, relax, threads, log, mip_rel_gap):
        import gurobipy as gp
        from gurobipy import GRB
        mdl = gp.Model()
        mdl.setParam("OutputFlag", int(bool(log)))
        mdl.setParam("TimeLimit", float(time_limit))
        mdl.setParam("Threads", int(threads))
        mdl.setParam("MIPGap", mip_rel_gap)
        vt = [GRB.BINARY if (b and not relax and lo == 0 and up == 1) else
              (GRB.INTEGER if (b and not relax) else GRB.CONTINUOUS)
              for b, lo, up in zip(self.integ, self.lb, self.ub)]
        xs = mdl.addVars(len(self.lb), lb=self.lb, ub=self.ub, obj=self.cost, vtype=vt)
        mdl.ModelSense = GRB.MAXIMIZE
        for lo, up, idx, val in self.rows:
            e = gp.LinExpr(val, [xs[i] for i in idx])
            if lo is not None and up is not None and lo == up:
                mdl.addLConstr(e, GRB.EQUAL, lo)
            else:
                if lo is not None:
                    mdl.addLConstr(e, GRB.GREATER_EQUAL, lo)
                if up is not None:
                    mdl.addLConstr(e, GRB.LESS_EQUAL, up)
        t0 = time.time()
        mdl.optimize()
        wall = time.time() - t0
        status = {GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "Time limit reached",
                  GRB.INFEASIBLE: "Infeasible"}.get(mdl.Status, str(mdl.Status))
        x = None
        obj = None
        if mdl.SolCount > 0:
            x = np.array([xs[i].X for i in range(len(self.lb))])
            obj = mdl.ObjVal
        bound = obj if relax else (mdl.ObjBound if mdl.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None)
        return {"status": status, "obj": obj, "bound": bound, "time": wall, "x": x,
                "nodes": int(getattr(mdl, "NodeCount", 0) or 0)}

    def _solve_highs(self, time_limit, relax, threads, log, mip_rel_gap):
        import highspy
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(log))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("threads", int(threads))
        h.setOptionValue("mip_rel_gap", mip_rel_gap)
        inf = highspy.kHighsInf
        nv = len(self.lb)
        lp = highspy.HighsLp()
        lp.num_col_ = nv
        lp.num_row_ = len(self.rows)
        lp.col_cost_ = np.array(self.cost, dtype=float)
        lp.col_lower_ = np.array(self.lb, dtype=float)
        lp.col_upper_ = np.array(self.ub, dtype=float)
        lp.row_lower_ = np.array([r[0] if r[0] is not None else -inf for r in self.rows], dtype=float)
        lp.row_upper_ = np.array([r[1] if r[1] is not None else inf for r in self.rows], dtype=float)
        starts, index, value = [], [], []
        # build column-wise matrix
        cols = [[] for _ in range(nv)]
        for ri, (_, _, idx, val) in enumerate(self.rows):
            for c, v in zip(idx, val):
                cols[c].append((ri, v))
        for c in range(nv):
            starts.append(len(index))
            for ri, v in cols[c]:
                index.append(ri); value.append(v)
        starts.append(len(index))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.array(starts, dtype=np.int32)
        lp.a_matrix_.index_ = np.array(index, dtype=np.int32)
        lp.a_matrix_.value_ = np.array(value, dtype=float)
        lp.sense_ = highspy.ObjSense.kMaximize
        if not relax:
            lp.integrality_ = [highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous
                               for b in self.integ]
        h.passModel(lp)
        t0 = time.time()
        h.run()
        wall = time.time() - t0
        info = h.getInfo()
        status = h.modelStatusToString(h.getModelStatus())
        sol = h.getSolution()
        x = np.array(sol.col_value) if sol.value_valid else None
        if relax:
            obj = info.objective_function_value if sol.value_valid else None
            bound = obj
        else:
            obj = info.objective_function_value if info.primal_solution_status >= 2 else None
            bound = info.mip_dual_bound
        return {"status": status, "obj": obj, "bound": bound, "time": wall, "x": x,
                "nodes": getattr(info, "mip_node_count", 0)}


def build_ti(inst: Instance):
    n, m, D0, Db = inst.n, inst.m, inst.D0, inst.Dbar
    B = _Builder()
    x = {}   # (j,h,t) -> var
    for j, tg in enumerate(inst.targets):
        for h, md in enumerate(tg.modes):
            for t in range(0, Db - md.p + 1):
                x[(j, h, t)] = B.var(0, 1, float(tg.w), True)
    # each target at most once
    for j, tg in enumerate(inst.targets):
        idx = [x[(j, h, t)] for h, md in enumerate(tg.modes) for t in range(0, Db - md.p + 1)]
        if idx:
            B.row(None, 1, idx, [1] * len(idx))
    # battery capacity at each period
    for i in range(m):
        for t in range(Db):
            idx = []
            for j, tg in enumerate(inst.targets):
                for h, md in enumerate(tg.modes):
                    if i in md.batteries:
                        for s in range(max(0, t - md.p + 1), min(t, Db - md.p) + 1):
                            idx.append(x[(j, h, s)])
            if idx:
                B.row(None, 1, idx, [1] * len(idx))
    givers = [j for j in range(n) if inst.targets[j].delta > 0]
    if Db > D0:
        # E_t = credit from completions < t ; E_{t} = E_{t-1} + sum_j delta_j * [C_j = t-1]
        E = {}
        prevE = None
        for t in range(1, Db + 1):
            E[t] = B.var(0, float(sum(inst.targets[j].delta for j in givers)))
            idx, val = [E[t]], [1.0]
            if prevE is not None:
                idx.append(prevE); val.append(-1.0)
            for j in givers:
                for h, md in enumerate(inst.targets[j].modes):
                    s = t - 1 - md.p
                    if 0 <= s <= Db - md.p:
                        idx.append(x[(j, h, s)]); val.append(-float(inst.targets[j].delta))
            B.row(None, 0, idx, val)    # E_t <= E_{t-1} + new credits
            prevE = E[t]
        # z_kt >= fraction of target k completing at or after t (cumulative, sparse); a_t >= z_kt;
        # (t-D0) a_t <= E_t   (same linear relaxation as summing the start variables directly)
        a = {t: B.var(0, 1) for t in range(D0 + 1, Db + 1)}
        for k, tg in enumerate(inst.targets):
            znext = None
            for t in range(Db, D0, -1):
                z = B.var(0, 1)
                idx, val = [z], [1.0]
                if znext is not None:
                    idx.append(znext); val.append(-1.0)
                for h, md in enumerate(tg.modes):
                    s0 = t - md.p                      # start such that the engagement ends at t
                    if 0 <= s0 <= Db - md.p:
                        idx.append(x[(k, h, s0)]); val.append(-1.0)
                B.row(0, None, idx, val)               # z_kt >= z_k,t+1 + [k ends at t]
                B.row(0, None, [a[t], z], [1.0, -1.0])  # a_t >= z_kt
                znext = z
        for t in range(D0 + 1, Db + 1):
            B.row(None, 0, [a[t], E[t]], [float(t - D0), -1.0])
    return B, x


def solve_ti(inst: Instance, time_limit: float = 60.0, relax: bool = False, threads: int = 1,
             log: bool = False, backend: str = "auto") -> dict:
    t0 = time.time()
    B, x = build_ti(inst)
    tb = time.time() - t0
    r = B.solve(time_limit, relax=relax, threads=threads, log=log, backend=backend)
    sol = {}
    if not relax and r["x"] is not None and r["obj"] is not None:
        for (j, h, t), v in x.items():
            if r["x"][v] > 0.5:
                sol[j] = (h, t)
    r.update({"sol": sol, "build_time": tb, "nvars": len(B.lb), "nrows": len(B.rows)})
    r.pop("x", None)
    return r


def build_dj(inst: Instance):
    n, m, D0, Db = inst.n, inst.m, inst.D0, inst.Dbar
    B = _Builder()
    M = Db
    y = {}
    for j, tg in enumerate(inst.targets):
        for h, md in enumerate(tg.modes):
            y[(j, h)] = B.var(0, 1, float(tg.w), True)
    S = [B.var(0, Db, 0.0) for _ in range(n)]
    C = [B.var(0, Db, 0.0) for _ in range(n)]
    for j, tg in enumerate(inst.targets):
        B.row(None, 1, [y[(j, h)] for h in range(len(tg.modes))], [1] * len(tg.modes))
        # C_j = S_j + sum_h p_jh y_jh
        B.row(0, 0, [C[j], S[j]] + [y[(j, h)] for h in range(len(tg.modes))],
              [1.0, -1.0] + [-float(md.p) for md in tg.modes])
    # u_ij = sum of y over modes of j using battery i
    def u(i, j):
        return [(y[(j, h)], 1.0) for h, md in enumerate(inst.targets[j].modes) if i in md.batteries]
    for j in range(n):
        for k in range(j + 1, n):
            share = [i for i in range(m) if u(i, j) and u(i, k)]
            if not share:
                continue
            o = B.var(0, 1, 0.0, True)   # 1 if j before k
            for i in share:
                uj, uk = u(i, j), u(i, k)
                # C_j <= S_k + M(1-o) + M(2 - u_ij - u_ik)
                idx = [C[j], S[k], o] + [v for v, _ in uj] + [v for v, _ in uk]
                val = [1.0, -1.0, float(M)] + [float(M)] * len(uj) + [float(M)] * len(uk)
                B.row(None, 3 * M, idx, val)
                # C_k <= S_j + M o + M(2 - u_ij - u_ik)
                idx = [C[k], S[j], o] + [v for v, _ in uj] + [v for v, _ in uk]
                val = [1.0, -1.0, -float(M)] + [float(M)] * len(uj) + [float(M)] * len(uk)
                B.row(None, 2 * M, idx, val)
    givers = [j for j in range(n) if inst.targets[j].delta > 0]
    for k, tg in enumerate(inst.targets):
        yk = [y[(k, h)] for h in range(len(tg.modes))]
        idx, val = [C[k]] + yk, [1.0] + [float(M)] * len(yk)
        for j in givers:
            if j == k:
                continue
            z = B.var(0, 1, 0.0, True)
            B.row(None, 0, [z] + [y[(j, h)] for h in range(len(inst.targets[j].modes))],
                  [1.0] + [-1.0] * len(inst.targets[j].modes))
            # C_j + 1 <= C_k + M (1 - z)
            B.row(None, M - 1, [C[j], C[k], z], [1.0, -1.0, float(M)])
            idx.append(z); val.append(-float(inst.targets[j].delta))
        B.row(None, D0 + M, idx, val)   # C_k <= D0 + sum delta z + M(1 - sum y_k)
    return B, y, S


def solve_dj(inst: Instance, time_limit: float = 60.0, relax: bool = False, threads: int = 1,
             log: bool = False, backend: str = "auto") -> dict:
    t0 = time.time()
    B, y, S = build_dj(inst)
    tb = time.time() - t0
    r = B.solve(time_limit, relax=relax, threads=threads, log=log, backend=backend)
    sol = {}
    if not relax and r["x"] is not None and r["obj"] is not None:
        for (j, h), v in y.items():
            if r["x"][v] > 0.5:
                sol[j] = (h, int(np.floor(r["x"][S[j]] + 1e-6)))
    r.update({"sol": sol, "build_time": tb, "nvars": len(B.lb), "nrows": len(B.rows)})
    r.pop("x", None)
    return r
