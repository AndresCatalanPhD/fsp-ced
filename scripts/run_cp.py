"""CP models, IG+CP hybrid and logic-based Benders decomposition (LBBD) on every instance and variant.

python scripts/run_cp.py --inst instances --out results/cp.jsonl --tl 300 [--workers 2]
                         [--methods CP,CPF,IG+CP,LBBD] [--ig-time 5] [--shard 0/1] [--variants CD,ED]

Monolithic models
  CP      CP model with pairwise crediting literals (the natural model; CP-P in the paper)
  CPF     flow-shop CP model: one second-stage interval per extending target, no crediting literals (CP-F), with
          the caps, load bounds and ladder inequalities as redundant constraints
  CPF0    the flow-shop CP model without these redundant constraints
  CPS     the pairwise CP model with these redundant constraints
  IG+CPF  IG-ED for --ig-time seconds, then CP-F warm-started with its plan (same start as LBBD)
  IG+CP   IG-ED for --ig-time seconds, then the CP model warm-started
Decomposition (fspced/lbbd.py, version 3)
  LBBD    Johnson and ladder inequalities in the master problem, flow-shop subproblem, deficit cuts
Ablation (write these to a separate file, e.g. results/ablation.jsonl, because LBBD-S can certify wrong bounds)
  LBBD-B  basic master problem: no Johnson and no ladder inequalities (battery loads only)
  LBBD-P  pairwise CP model in the subproblem instead of the flow-shop model
  LBBD-N  no-good cuts: every deficit and overrun replaced by 1 (lifted no-good cuts, still valid)
  LBBD-S  classical no-good cuts (NOT valid with an expandable deadline; kept to measure the flaw)
  LBBD-T  subproblem in the space of targets: the modes of the master problem are not passed to the subproblem
  LBBD-BN basic master problem and no-good cuts (isolates the effect of the deficits when the master is weak)
  LBBD-0  conventional design: basic master, targets only, pairwise subproblem and lifted no-good cuts
"""
import argparse, os, sys
from common import instance_files, done_keys, append, budget_ok
from fspced.instance import Instance
from fspced.cpsat import solve_cp
from fspced.flowshop import solve_cpf
from fspced.heur import ig_cp, ig_cpf
from fspced.lbbd import solve_lbbd
from fspced.check import check

LBBD = {"LBBD": {}, "LBBD-B": {"johnson": False}, "LBBD-P": {"sub": "pair"}, "LBBD-N": {"cuts": "nogood"},
        "LBBD-S": {"cuts": "standard"}, "LBBD-T": {"fix_modes": False},
        "LBBD-BN": {"johnson": False, "cuts": "nogood"},
        "LBBD-0": {"johnson": False, "sub": "pair", "cuts": "nogood", "fix_modes": False}}
EXTRA = ["subproblems", "fixed_hits", "free_hits", "johnson_cuts", "johnson_fails", "overrun_cuts", "deficit_cuts",
         "theta_sum", "t_master", "t_sub", "sub", "johnson", "fix_modes", "valid"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="instances"); ap.add_argument("--out", default="results/cp.jsonl")
    ap.add_argument("--tl", type=float, default=300); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--methods", default="CP,CPF,IG+CP,LBBD"); ap.add_argument("--ig-time", type=float, default=5.0)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--sub-limit", type=float, default=10.0)
    ap.add_argument("--variants", default="CD,ED"); ap.add_argument("--master", default="auto")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--pick", default="")   # quick tests only
    a = ap.parse_args()
    done = done_keys(a.out, ["instance", "variant", "method"])
    for f in instance_files(a.inst, a.shard, limit=a.limit, pick=a.pick):
        base = Instance.load(f)
        name = os.path.basename(f)[:-5]
        for var in a.variants.split(","):
            inst = base.common_variant() if var == "CD" else base
            for meth in a.methods.split(","):
                if (name, var, meth) in done:
                    continue
                if not budget_ok():
                    sys.exit(0)
                if meth == "CP":
                    r = solve_cp(inst, a.tl, workers=a.workers, model="pair")
                elif meth == "CPF":
                    r = solve_cpf(inst, a.tl, workers=a.workers)
                elif meth == "CPF0":
                    r = solve_cpf(inst, a.tl, workers=a.workers, strengthen=False)
                elif meth == "CPS":
                    r = solve_cp(inst, a.tl, workers=a.workers, model="pair", strengthen=True)
                elif meth in LBBD:
                    r = solve_lbbd(inst, a.tl, ig_time=a.ig_time, workers=a.workers, sub_limit=a.sub_limit,
                                   master=a.master, **LBBD[meth])
                elif meth == "IG+CPF":
                    r = ig_cpf(inst, a.tl, ig_time=a.ig_time, workers=a.workers)
                elif meth == "IG+CP":
                    r = ig_cp(inst, a.tl, ig_time=a.ig_time, workers=a.workers)
                else:
                    raise SystemExit(f"unknown method {meth}")
                ok, val, msg = check(inst, r["sol"]) if r["sol"] else (True, 0, "")
                rec = {"instance": name, "variant": var, "method": meth, "value": r["value"],
                       "bound": r["bound"], "optimal": bool(r["optimal"]), "time": r["time"],
                       "status": r["status"], "check": ok, "check_value": val, "tl": a.tl,
                       "ig_value": r.get("ig_value"), "iters": r.get("iters"), "cuts": r.get("cuts"),
                       "unknown": r.get("unknown"), "decoder_hits": r.get("decoder_hits"),
                       "master": r.get("master"), "cuts_type": r.get("cuts_type"), "fallback": r.get("fallback")}
                rec.update({k: r.get(k) for k in EXTRA})
                rec["sol"] = {str(j): v for j, v in r["sol"].items()}
                append(a.out, rec)
                print(name, var, meth, r["value"], round(r["bound"], 1), r["status"], round(r["time"], 1), ok,
                      r.get("iters"), flush=True)
