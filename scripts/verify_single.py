"""Numerical check of Theorem 2 (single battery): Johnson-order DP = brute force = CP-SAT.
python scripts/verify_single.py 300"""
import sys, itertools, random
from common import ROOT
from fspced.instance import Instance, Target, Mode
from fspced.single import solve_dp, feasible_set
from fspced.cpsat import solve_cp

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    random.seed(7)
    mism = 0
    for it in range(N):
        n = random.randint(4, 9)
        tg = []
        for j in range(n):
            p = random.randint(1, 12); w = random.randint(1, 10)
            dl = random.randint(0, 12) if random.random() < 0.6 else 0
            tg.append(Target(w, dl, [Mode((0,), p)]))
        D0 = random.randint(5, 30); Db = D0 + random.randint(0, 25)
        inst = Instance(f"s{it}", 1, D0, Db, tg)
        v_dp = solve_dp(inst)
        v_bf = max(sum(tg[j].w for j in S) for r in range(n + 1) for S in itertools.combinations(range(n), r)
                   if feasible_set(inst, S))
        v_cp = solve_cp(inst, 20, workers=1)["value"]
        if not (v_dp == v_bf == v_cp):
            mism += 1
            print("MISMATCH", it, v_dp, v_bf, v_cp)
    print(f"single-battery check: {N} random instances, {mism} mismatches")
