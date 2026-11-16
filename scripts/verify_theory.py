"""Numerical checks of the structural results of the paper on small random instances.

  A. Flow-shop model (Proposition 3 and Lemma 2): the flow-shop CP model and the pairwise CP model give the same
     optimal value, and every plan passes the independent checker.
  B. Master problem (Theorem 5): its optimal value is an upper bound on the optimum; it is exact for one battery
     and, with a common deadline, for two batteries or single-battery modes.
  C. Deficit inequalities (Theorem 6): for random certificates (core, helpers, allowed modes, theta computed by
     the relaxation), every feasible plan of the instance, found by complete enumeration of the selections and
     modes, satisfies the inequality; the classical cut (no outside terms) is counted when it is violated.
  D. Decomposition: LBBD without the incumbent heuristic returns the optimum with deficit cuts and with lifted
     no-good cuts.

python scripts/verify_theory.py [--n 200] [--seed 5] [--parts ABCD] [--out results/theory.jsonl]
With --out, the counts are appended to a JSONL file read by scripts/analyze.py (macros of the paper).
"""
import argparse, itertools, random, sys
from common import ROOT, append  # noqa: F401
from fspced.instance import Instance, Target, Mode, prune_dominated
from fspced.cpsat import solve_cp
from fspced.check import check
from fspced import flowshop as fs
from fspced.lbbd import Master, deficit_cut, johnson_rows, johnson_violated, master_rows, solve_lbbd, givers_of


def rand_inst(rng, name, m=None, n=None, single=False, cd=False, one_mode=False):
    m = m or rng.choice([1, 2, 2, 3, 3, 4])
    n = n or rng.randint(4, 8)
    D0 = rng.randint(4, 14)
    Db = D0 if cd else D0 + rng.randint(0, 16)
    tg = []
    for _ in range(n):
        modes = []
        for _ in range(1 if one_mode else rng.choice([1, 2, 2, 3])):
            q = 1 if single else min(m, rng.choice([1, 1, 2, 2, 3]))
            modes.append(Mode(tuple(sorted(rng.sample(range(m), q))), rng.randint(1, D0 + 3)))
        d = 0 if cd else rng.choice([0, 0, rng.randint(1, 8), rng.randint(1, 4)])
        tg.append(Target(rng.randint(1, 20), d, prune_dominated(modes)))
    return Instance(name, m, D0, Db, tg)


def part_a(N, rng):
    bad = 0
    for k in range(N):
        inst = rand_inst(rng, f"a{k}")
        a = solve_cp(inst, 20, model="pair")
        b = fs.solve_cpf(inst, 20, strengthen=bool(k % 2))
        okb = check(inst, b["sol"])[0] if b["sol"] else True
        if not (a["optimal"] and b["optimal"] and a["value"] == b["value"] and okb):
            bad += 1
            print("A MISMATCH", k, a["value"], b["value"], okb)
    print(f"A. flow-shop model vs pairwise model: {N} instances, {bad} mismatches", flush=True)
    return bad, {"A_n": N, "A_bad": bad}


def part_b(N, rng):
    bad = exact_bad = n_exact = 0
    for k in range(N):
        kind = k % 4
        if kind == 0:
            inst = rand_inst(rng, f"b{k}", m=1, one_mode=True)                 # one battery
        elif kind == 1:
            inst = rand_inst(rng, f"b{k}", m=2, cd=True)                       # CD, two batteries
        elif kind == 2:
            inst = rand_inst(rng, f"b{k}", m=rng.choice([3, 4]), single=True, cd=True)   # CD, single-battery modes
        else:
            inst = rand_inst(rng, f"b{k}")
        ref = solve_cp(inst, 20, model="pair")
        M = Master(inst, backend="cpsat", threads=1)
        if k % 2:                         # all Johnson inequalities from the start, or added when violated
            M.cuts.extend(johnson_rows(inst, M.R))
        done = set()
        while True:
            st, S, modes, bound = M.solve(20)
            bad_f = [f for f in (johnson_violated(inst, M.R, S, modes) if S else []) if f not in done]
            if k % 2:
                assert not bad_f
            if not bad_f:
                break
            M.cuts.extend(johnson_rows(inst, M.R, set(bad_f)))
            done.update(bad_f)
        mv = sum(inst.targets[j].w for j in S) if S else 0
        if mv < ref["value"]:
            bad += 1
            print("B NOT A RELAXATION", k, mv, ref["value"])
        if kind < 3:
            n_exact += 1
            if mv != ref["value"]:
                exact_bad += 1
                print("B NOT EXACT", k, kind, mv, ref["value"])
    print(f"B. master problem: {N} instances, {bad} below the optimum; exact cases: {n_exact}, {exact_bad} failures",
          flush=True)
    return bad + exact_bad, {"B_n": N, "B_bad": bad, "B_exact_n": n_exact, "B_exact_bad": exact_bad}


def feasible_configs(inst):
    """All (selection, modes) that can be engaged, by complete enumeration (small instances only)."""
    n = inst.n
    out = []
    choices = [[None] + list(range(len(t.modes))) for t in inst.targets]
    for cfg in itertools.product(*choices):
        S = [j for j in range(n) if cfg[j] is not None]
        if not S:
            out.append(cfg)
            continue
        modes = {j: cfg[j] for j in S}
        st, _ = fs.check(inst, S, 10, modes=modes, fixed=True, core=False)
        if st == "feasible":
            out.append(cfg)
    return out


def part_c(N, rng):
    bad = n_cert = classical_viol = classical_n = strict = 0
    for k in range(N):
        inst = rand_inst(rng, f"c{k}", n=rng.randint(4, 6))
        if inst.n_modes() > 12:
            continue
        feas = feasible_configs(inst)
        R = master_rows(inst)
        G = set(givers_of(inst))
        for _ in range(6):
            fixed = rng.random() < 0.5
            size = rng.randint(1, min(4, inst.n))
            ok_t = [j for j in range(inst.n) if any((j, h) in R["xidx"] for h in range(len(inst.targets[j].modes)))]
            if not ok_t:
                continue
            U = rng.sample(ok_t, min(size, len(ok_t)))
            size = len(U)
            C = U[:rng.randint(1, size)]
            helpers = U[len(C):]
            modes = {j: rng.choice([h for h in range(len(inst.targets[j].modes)) if (j, h) in R["xidx"]]) for j in U}
            allowed = {j: [modes[j]] for j in U} if fixed else None
            th = fs.relaxation(inst, C, 10, allowed=allowed, helpers=helpers)
            ov = fs.relaxation(inst, C, 10, allowed=allowed, cap_only=True)
            for theta, outside in ((th, True), (ov, False)):
                if theta < 1:
                    continue
                n_cert += 1
                strict += theta > 1
                c, rhs = deficit_cut(inst, R, C, helpers, modes, fixed, theta, outside=outside)
                for cfg in feas:
                    lhs = 0.0
                    for j in range(inst.n):
                        if cfg[j] is not None:
                            lhs += c.get(R["yidx"][j], 0.0) + c.get(R["xidx"].get((j, cfg[j]), -1), 0.0)
                    if lhs < rhs - 1e-9:
                        bad += 1
                        print("C VIOLATED", k, C, helpers, fixed, theta, outside, cfg)
                        break
            if th >= 1 and ov < 1:      # the classical cut drops the outside terms without an overrun certificate
                classical_n += 1
                c, rhs = deficit_cut(inst, R, C, helpers, modes, fixed, 1, outside=False)
                for cfg in feas:
                    lhs = sum(c.get(R["yidx"][j], 0.0) + c.get(R["xidx"].get((j, cfg[j]), -1), 0.0)
                              for j in range(inst.n) if cfg[j] is not None)
                    if lhs < rhs - 1e-9:
                        classical_viol += 1
                        break
    print(f"C. deficit inequalities: {n_cert} certificates ({strict} with theta > 1), {bad} violated by a feasible "
          f"plan; classical cut violated in {classical_viol} of {classical_n} certificates without overrun",
          flush=True)
    return bad, {"C_cert": n_cert, "C_strict": strict, "C_bad": bad, "C_classical_n": classical_n,
                 "C_classical_bad": classical_viol}


def part_d(N, rng):
    bad = wrong_std = 0
    its = {"deficit": 0, "nogood": 0, "standard": 0}
    for k in range(N):
        inst = rand_inst(rng, f"d{k}", n=rng.randint(6, 10))
        ref = solve_cp(inst, 30, model="pair")
        for cuts in ("deficit", "nogood", "standard"):
            r = solve_lbbd(inst, 30, ig_time=0, workers=1, cuts=cuts, fallback=False, master="cpsat")
            its[cuts] += r["iters"]
            if cuts == "standard":
                wrong_std += r["optimal"] and r["value"] != ref["value"]
            elif not (r["optimal"] and r["value"] == ref["value"]):
                bad += 1
                print("D MISMATCH", k, cuts, r["value"], ref["value"], r["status"])
    print(f"D. decomposition: {N} instances, {bad} mismatches with valid cuts; wrong optima certified with the "
          f"classical cut: {wrong_std}", flush=True)
    return bad, {"D_n": N, "D_bad": bad, "D_wrong_std": int(wrong_std), "D_iters_deficit": its["deficit"],
                 "D_iters_nogood": its["nogood"], "D_iters_standard": its["standard"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--parts", default="ABCD"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    fails = 0
    rec = {"instance": f"random_n{a.n}_seed{a.seed}", "variant": "ED", "method": "THEORY", "parts": a.parts}
    for part, fn in (("A", part_a), ("B", part_b), ("C", part_c), ("D", part_d)):
        if part in a.parts:
            f, st = fn(a.n, random.Random(a.seed * 1000 + ord(part)))
            fails += f
            rec.update(st)
    rec["failures"] = fails
    if a.out:
        append(a.out, rec)
    print("THEORY CHECK OK" if fails == 0 else f"THEORY CHECK FAILED ({fails})")
    sys.exit(0 if fails == 0 else 1)
