"""Strength of the master problem: linear and integer bounds with and without the inequalities of Section 4.

python scripts/root_bounds.py --inst instances --out results/root.jsonl [--tl 60] [--shard 0/1]

For every instance (ED variant) it records the optimal value of
  lp0 / mip0   master problem with the load bounds only (variant B of the ablation),
  lp1 / mip1   + ladder inequalities,
  lp2 / mip2   + all Johnson inequalities of the battery families, triangle families and pooled batteries
               (the master problem of LBBD before any Benders cut),
as linear relaxation (lp*) and as integer program (mip*), solved with HiGHS (scipy).  scripts/analyze.py compares
them with the best known value of each instance.  Deterministic; a few minutes per instance set.
"""
import argparse, os, sys, time
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from common import instance_files, done_keys, append, budget_ok
from fspced.instance import Instance
from fspced.lbbd import master_rows, johnson_rows

INF = 1e20


def build(inst, ladder, johnson):
    R = master_rows(inst, ladder)
    rows, lo, up = list(R["rows"]), list(R["lo"]), list(R["up"])
    if johnson:
        for c, rhs in johnson_rows(inst, R):
            rows.append(c); lo.append(rhs); up.append(INF)
    ri, ci, vv = [], [], []
    for r, c in enumerate(rows):
        for q, v in c.items():
            ri.append(r); ci.append(q); vv.append(v)
    A = coo_matrix((vv, (ri, ci)), shape=(len(rows), R["nv"])).tocsr()
    lo = np.array([max(-INF, x) for x in lo]); up = np.array([min(INF, x) for x in up])
    return R, A, lo, up


def solve(R, A, lo, up, integer, tl):
    c = -np.asarray(R["w"], dtype=float)
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A, lo, up), bounds=Bounds(0, 1),
               integrality=np.ones(R["nv"]) if integer else np.zeros(R["nv"]),
               options={"time_limit": tl, "disp": False})
    t = time.time() - t0
    if res.x is None:
        return None, t, res.status
    # for an integer program stopped by the time limit, report the dual bound (valid upper bound)
    val = -res.fun
    if integer and res.status != 0 and getattr(res, "mip_dual_bound", None) is not None:
        val = -res.mip_dual_bound
    return float(val), t, int(res.status)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="instances"); ap.add_argument("--out", default="results/root.jsonl")
    ap.add_argument("--tl", type=float, default=60.0); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--pick", default="")
    a = ap.parse_args()
    done = done_keys(a.out, ["instance", "variant", "method"])
    for f in instance_files(a.inst, a.shard, limit=a.limit, pick=a.pick):
        name = os.path.basename(f)[:-5]
        if (name, "ED", "ROOT") in done:
            continue
        if not budget_ok():
            sys.exit(0)
        inst = Instance.load(f)
        rec = {"instance": name, "variant": "ED", "method": "ROOT"}
        for tag, ladder, johnson in (("0", False, False), ("1", True, False), ("2", True, True)):
            R, A, lo, up = build(inst, ladder, johnson)
            rec["lp" + tag], rec["t_lp" + tag], _ = solve(R, A, lo, up, False, a.tl)
            rec["mip" + tag], rec["t_mip" + tag], rec["st_mip" + tag] = solve(R, A, lo, up, True, a.tl)
            rec["rows" + tag] = A.shape[0]
        append(a.out, rec)
        print(name, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in rec.items()
                     if k.startswith(("lp", "mip"))}, flush=True)
