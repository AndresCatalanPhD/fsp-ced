"""Time-indexed (TI) and disjunctive (DJ) MILPs and their LP relaxations (Gurobi if licensed, else HiGHS).
Run in a process that does NOT import OR-Tools.

python scripts/run_milp.py --inst instances --out results/milp.jsonl --tl 300 --nmax-ti 1000 --nmax-dj 20
"""
import argparse, os, sys
from common import instance_files, done_keys, append, budget_ok
from fspced.instance import Instance
from fspced.milp import solve_ti, solve_dj, default_backend
from fspced.check import check

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="instances"); ap.add_argument("--out", default="results/milp.jsonl")
    ap.add_argument("--tl", type=float, default=300); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tl-dj", type=float, default=60)   # the disjunctive MILP is only a baseline
    ap.add_argument("--nmax-ti", type=int, default=1000); ap.add_argument("--nmax-dj", type=int, default=20)
    ap.add_argument("--backend", default="auto"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--pick", default="")   # quick tests only
    ap.add_argument("--variants", default="CD,ED")
    a = ap.parse_args()
    backend = default_backend() if a.backend == "auto" else a.backend
    print("MILP backend:", backend, flush=True)
    done = done_keys(a.out, ["instance", "variant", "method"])
    for f in instance_files(a.inst, a.shard, limit=a.limit, pick=a.pick):
        base = Instance.load(f)
        name = os.path.basename(f)[:-5]
        for var in a.variants.split(","):
            inst = base.common_variant() if var == "CD" else base
            todo = [("TI-LP", solve_ti, True)]
            if inst.n <= a.nmax_ti:
                todo.append(("TI", solve_ti, False))
            if inst.n <= a.nmax_dj:
                todo += [("DJ-LP", solve_dj, True), ("DJ", solve_dj, False)]
            for meth, fn, relax in todo:
                if (name, var, meth) in done:
                    continue
                if not budget_ok():
                    sys.exit(0)
                tl = a.tl_dj if meth.startswith("DJ") else a.tl
                r = fn(inst, tl, relax=relax, threads=a.threads, backend=backend)
                ok, val = True, None
                if not relax and r["sol"]:
                    ok, val, _ = check(inst, r["sol"])
                append(a.out, {"instance": name, "variant": var, "method": meth,
                               "value": None if r["obj"] is None else (r["obj"] if relax else int(round(r["obj"]))),
                               "bound": r["bound"], "status": r["status"], "time": r["time"],
                               "build_time": r["build_time"], "nvars": r["nvars"], "nrows": r["nrows"],
                               "check": ok, "check_value": val, "backend": r.get("backend"), "tl": tl})
                print(name, var, meth, r["obj"], r["bound"], r["status"], round(r["time"], 1), ok, flush=True)
