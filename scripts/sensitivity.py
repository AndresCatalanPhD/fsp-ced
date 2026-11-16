"""Value of the expandable deadline under other caps and extensions (coastal set, ED variant, solved by LBBD).

python scripts/sensitivity.py --inst instances --out results/sensitivity.jsonl [--tl 120] [--shard 0/1]

Every instance is solved again with
  cap factor k: Dbar = ceil(k * D0), k in {1.25, 2.0}, or no cap (Dbar = D0 + total extension), extensions as given;
  extension scale s: delta_j = round(s * delta_j), s in {0.5, 2.0}, with the cap of the benchmark (Dbar = ceil(1.5 D0)).
The CD variant does not depend on k or s, so the gain of each configuration is measured against the CD optimum of the
main campaign (scripts/analyze.py).  Records: method SENS, variant 'k<k>_s<s>'.
"""
import argparse, copy, math, os, sys
from common import instance_files, done_keys, append, budget_ok
from fspced.instance import Instance
from fspced.lbbd import solve_lbbd
from fspced.check import check

CONFIGS = [(1.25, 1.0), (2.0, 1.0), (None, 1.0), (1.5, 0.5), (1.5, 2.0)]   # (cap factor or None = no cap, scale)


def variant(base, k, s):
    inst = copy.deepcopy(base)
    for t in inst.targets:
        t.delta = int(round(s * t.delta))
    inst.Dbar = inst.D0 + sum(t.delta for t in inst.targets) if k is None else int(math.ceil(k * inst.D0))
    inst.Dbar = max(inst.Dbar, inst.D0)
    return inst


def tag(k, s):
    return f"k{'inf' if k is None else k}_s{s}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="instances"); ap.add_argument("--out", default="results/sensitivity.jsonl")
    ap.add_argument("--tl", type=float, default=120.0); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pick", default="")
    a = ap.parse_args()
    done = done_keys(a.out, ["instance", "variant", "method"])
    for f in instance_files(a.inst, a.shard, limit=a.limit, pick=a.pick):
        name = os.path.basename(f)[:-5]
        base = Instance.load(f)
        for k, s in CONFIGS:
            v = tag(k, s)
            if (name, v, "SENS") in done:
                continue
            if not budget_ok():
                sys.exit(0)
            inst = variant(base, k, s)
            r = solve_lbbd(inst, a.tl, ig_time=2, workers=a.workers)
            ok, val, _ = check(inst, r["sol"]) if r["sol"] else (True, 0, "")
            append(a.out, {"instance": name, "variant": v, "method": "SENS", "cap": k, "scale": s,
                           "Dbar": inst.Dbar, "value": r["value"], "bound": r["bound"],
                           "optimal": bool(r["optimal"]), "time": r["time"], "check": ok,
                           "sol": {str(j): x for j, x in r["sol"].items()}})
            print(name, v, r["value"], round(r["bound"], 1), r["status"], round(r["time"], 1), ok, flush=True)
