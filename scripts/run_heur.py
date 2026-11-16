"""MWSPT dispatching rule, GA and IG-ED.

python scripts/run_heur.py --inst instances --out results/heur.jsonl --tl 10 --seeds 0,1,2 [--shard 0/1]
"""
import argparse, os, sys
from common import instance_files, done_keys, append, budget_ok
from fspced.instance import Instance
from fspced.heur import mwspt, ga, ig
from fspced.check import check

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="instances"); ap.add_argument("--out", default="results/heur.jsonl")
    ap.add_argument("--tl", type=float, default=10); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--pick", default="")   # quick tests only
    ap.add_argument("--variants", default="CD,ED")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    done = done_keys(a.out, ["instance", "variant", "method", "seed"])
    files = instance_files(a.inst, a.shard, by_size=False, limit=a.limit, pick=a.pick)
    ig(Instance.load(files[0]), 0.5)   # numba warm-up (compilation time is not measured)
    for f in files:
        base = Instance.load(f)
        name = os.path.basename(f)[:-5]
        for var in a.variants.split(","):
            inst = base.common_variant() if var == "CD" else base
            runs = [("MWSPT", 0)] + [("GA", s) for s in seeds] + [("IG", s) for s in seeds]
            for meth, s in runs:
                if (name, var, meth, s) in done:
                    continue
                if not budget_ok():
                    sys.exit(0)
                r = mwspt(inst) if meth == "MWSPT" else (ga(inst, a.tl, seed=s) if meth == "GA" else ig(inst, a.tl, seed=s))
                ok, val, msg = check(inst, r["sol"])
                append(a.out, {"instance": name, "variant": var, "method": meth, "seed": s, "value": r["value"],
                               "time": r["time"], "t_best": r.get("t_best", r["time"]), "iters": r.get("iters"),
                               "init": r.get("init"), "evals": r.get("evals"), "tl": a.tl,
                               "check": bool(ok and val == r["value"]),
                               "sol": {str(j): v for j, v in r["sol"].items()}})
                print(name, var, meth, s, r["value"], round(r.get("t_best", 0), 2), ok, flush=True)
