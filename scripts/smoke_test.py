"""Quick end-to-end test of every runner before the real campaign (about 15 minutes on Colab).

Runs each experiment script on a few small instances (and on the smallest large instance) with short time limits, writing
to a scratch folder that is NOT the results folder, then runs the analysis on those records and checks that

  * every script finished without error,
  * every reported plan passed the independent checker,
  * the decomposition records carry the current code version,
  * the stress run reached the code paths for undecided subproblems (cut without proof, fallback to CP-F),
  * the structural results of the paper hold on a few random instances (scripts/verify_theory.py),
  * the analysis produced the macros and tables of the paper.

python scripts/smoke_test.py [--out smoke] [--workers 2]
Prints "SMOKE TEST OK" at the end, or the list of failures.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import version_of  # noqa: E402


def run(step, args, log, fails):
    t0 = time.time()
    cmd = [sys.executable, os.path.join(HERE, args[0])] + [str(x) for x in args[1:]]
    with open(log, "a") as fo:
        fo.write(f"\n### {step}: {' '.join(cmd)}\n")
        fo.flush()
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fo, stderr=subprocess.STDOUT)
    status = "ok" if rc == 0 else f"FAILED (exit code {rc}, see {log})"
    print(f"  {step:<44s} {time.time() - t0:6.1f} s  {status}", flush=True)
    if rc != 0:
        fails.append(step)


def records(path):
    out = []
    for f in glob.glob(path):
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoke"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    log = os.path.join(out, "smoke.log")
    W = a.workers
    fails = []
    print("Smoke test (short limits, scratch folder %s):" % out, flush=True)
    # main benchmark: the two smallest instances, every method
    run("heuristics (MWSPT, GA, IG-ED)", ["run_heur.py", "--inst", "instances", "--out", f"{out}/heur.jsonl",
                                           "--tl", 2, "--seeds", "0", "--limit", 2], log, fails)
    run("MILPs (TI, DJ, LP relaxations)", ["run_milp.py", "--inst", "instances", "--out", f"{out}/milp.jsonl",
                                            "--tl", 20, "--tl-dj", 10, "--threads", W, "--limit", 2], log, fails)
    run("CP-P, CP-F, warm starts, LBBD", ["run_cp.py", "--inst", "instances", "--out", f"{out}/cp.jsonl", "--tl", 20,
                                    "--workers", W, "--ig-time", 2, "--methods", "CP,CPF,IG+CP,IG+CPF,LBBD,CPS,CPF0", "--limit", 2],
        log, fails)
    run("ablation of the decomposition", ["run_cp.py", "--inst", "instances", "--out", f"{out}/ablation.jsonl",
                                          "--tl", 20, "--workers", W, "--ig-time", 2, "--methods",
                                          "LBBD-T,LBBD-P,LBBD-B,LBBD-N,LBBD-BN,LBBD-0,LBBD-S", "--variants", "ED", "--limit", 1],
        log, fails)
    # stress: subproblems left undecided on purpose (cut without proof, loss of the certificate, fallback to CP-F)
    run("LBBD stress (undecided subproblems)", ["run_cp.py", "--inst", "instances", "--out", f"{out}/stress.jsonl",
                                                "--tl", 30, "--workers", W, "--ig-time", 1, "--methods", "LBBD",
                                                "--variants", "ED", "--pick", "m6_n40_t30_r1", "--sub-limit", 0],
        log, fails)
    # Cha-Kim design: one instance
    run("Cha-Kim: heuristics", ["run_heur.py", "--inst", "instances_chakim", "--out", f"{out}/ck_heur.jsonl",
                                "--tl", 2, "--seeds", "0", "--limit", 1], log, fails)
    run("Cha-Kim: TI MILP", ["run_milp.py", "--inst", "instances_chakim", "--out", f"{out}/ck_milp.jsonl",
                             "--tl", 20, "--threads", W, "--nmax-ti", 1000, "--nmax-dj", 0, "--limit", 1], log, fails)
    run("Cha-Kim: CP-P, CP-F, LBBD", ["run_cp.py", "--inst", "instances_chakim", "--out", f"{out}/ck_cp.jsonl",
                                      "--tl", 20, "--workers", W, "--ig-time", 2, "--methods", "CP,CPF,LBBD",
                                      "--limit", 1], log, fails)
    # short-window set: the two smallest instances (ED only)
    run("short-window: heuristics", ["run_heur.py", "--inst", "instances_short", "--out", f"{out}/short_heur.jsonl",
                                     "--tl", 2, "--seeds", "0", "--variants", "ED", "--limit", 2], log, fails)
    run("short-window: TI MILP", ["run_milp.py", "--inst", "instances_short", "--out", f"{out}/short_milp.jsonl",
                                  "--tl", 20, "--threads", W, "--nmax-ti", 1000, "--nmax-dj", 0, "--variants", "ED",
                                  "--limit", 2], log, fails)
    run("short-window: CP-P, CP-F, LBBD", ["run_cp.py", "--inst", "instances_short", "--out", f"{out}/short_cp.jsonl",
                                           "--tl", 20, "--workers", W, "--ig-time", 2, "--methods", "CP,CPF,LBBD",
                                           "--variants", "ED", "--limit", 2], log, fails)
    run("short-window: ablation", ["run_cp.py", "--inst", "instances_short", "--out", f"{out}/short_ablation.jsonl",
                                   "--tl", 20, "--workers", W, "--ig-time", 2, "--methods",
                                   "LBBD-T,LBBD-P,LBBD-B,LBBD-N,LBBD-BN,LBBD-0,LBBD-S", "--variants", "ED", "--limit", 2], log, fails)
    # the smallest large instance (200 targets): memory and time at full size
    run("large instance: CP-F, LBBD", ["run_cp.py", "--inst", "instances_large", "--out", f"{out}/large_cp.jsonl",
                                       "--tl", 40, "--workers", W, "--ig-time", 2, "--methods", "CPF,LBBD",
                                       "--variants", "ED", "--limit", 1], log, fails)
    run("large instance: heuristics", ["run_heur.py", "--inst", "instances_large", "--out", f"{out}/large_heur.jsonl",
                                       "--tl", 3, "--seeds", "0", "--limit", 1], log, fails)
    # dispersed set (arbitrary battery sets): one instance with a capped window
    run("dispersed: heuristics", ["run_heur.py", "--inst", "instances_dispersed", "--out", f"{out}/disp_heur.jsonl",
                                  "--tl", 2, "--seeds", "0", "--variants", "ED", "--limit", 1], log, fails)
    run("dispersed: CP-P, CP-F, warm starts, LBBD", ["run_cp.py", "--inst", "instances_dispersed", "--out",
                                                     f"{out}/disp_cp.jsonl", "--tl", 20, "--workers", W, "--ig-time", 2,
                                                     "--methods", "CP,CPF,IG+CP,IG+CPF,LBBD", "--variants", "ED",
                                                     "--limit", 1], log, fails)
    run("dispersed: root bounds", ["root_bounds.py", "--inst", "instances_dispersed", "--out",
                                   f"{out}/disp_root.jsonl", "--tl", 10, "--limit", 1], log, fails)
    # revision experiments: strength of the master problem and sensitivity of the gain
    run("root bounds of the master problem", ["root_bounds.py", "--inst", "instances", "--out", f"{out}/root.jsonl",
                                              "--tl", 10, "--limit", 2], log, fails)
    run("short-window: root bounds", ["root_bounds.py", "--inst", "instances_short", "--out",
                                      f"{out}/short_root.jsonl", "--tl", 10, "--limit", 2], log, fails)
    run("sensitivity to the cap and the extensions", ["sensitivity.py", "--inst", "instances", "--out",
                                                      f"{out}/sensitivity.jsonl", "--tl", 10, "--workers", W,
                                                      "--limit", 2], log, fails)
    run("structural results on random instances", ["verify_theory.py", "--n", 8, "--out", f"{out}/theory.jsonl"],
        log, fails)
    run("analysis (tables, figure and macros)", ["analyze.py", "--res", out, "--out", f"{out}/paper_results"],
        log, fails)

    # ------------------------------------------------------------------ checks on the records
    recs = records(os.path.join(out, "*.jsonl"))
    bad = [r for r in recs if r.get("check") is False]
    if bad:
        fails.append(f"{len(bad)} plans failed the checker")
    old = [r for r in recs if r.get("method", "").startswith(("LBBD", "CPF"))
           and str(r.get("code_version")) != version_of(r["method"])]
    if old:
        fails.append(f"{len(old)} records with an old code version")
    need = {"heur.jsonl": "MWSPT", "milp.jsonl": "TI", "cp.jsonl": "LBBD", "ck_cp.jsonl": "CPF",
            "ablation.jsonl": "LBBD-0", "short_cp.jsonl": "LBBD", "short_ablation.jsonl": "LBBD-N",
            "short_heur.jsonl": "IG", "short_milp.jsonl": "TI", "large_cp.jsonl": "LBBD", "large_heur.jsonl": "IG",
            "theory.jsonl": "THEORY", "disp_cp.jsonl": "LBBD", "disp_root.jsonl": "ROOT", "root.jsonl": "ROOT", "short_root.jsonl": "ROOT", "sensitivity.jsonl": "SENS"}
    for f, meth in need.items():
        if not any(r.get("method") == meth for r in records(os.path.join(out, f))):
            fails.append(f"no {meth} record in {f}")
    th = [r for r in records(os.path.join(out, "theory.jsonl")) if r.get("method") == "THEORY"]
    if th and th[-1].get("failures"):
        fails.append("the check of the structural results failed")
    stress = [r for r in records(os.path.join(out, "stress.jsonl")) if r.get("method") == "LBBD"]
    if not stress:
        fails.append("no record in stress.jsonl")
    elif not any((r.get("unknown") or 0) > 0 and (r.get("fallback") or 0) > 0 for r in stress):
        fails.append("the stress run did not reach the fallback")
    if not any(r.get("method") == "IG+CPF" for r in records(os.path.join(out, "cp.jsonl"))):
        fails.append("no IG+CPF record in cp.jsonl")
    for f in ("numbers.tex", "tab_exact.tex", "tab_ablation.tex", "tab_sets.tex", "tab_root.tex", "tab_warm.tex",
              "tab_disp.tex", "tab_bysize.tex",
              "Figure_3.pdf"):
        if not os.path.exists(os.path.join(out, "paper_results", f)):
            fails.append(f"analysis did not write {f}")
    n_rec = len(recs)
    print(f"  records written: {n_rec}; checker failures: {len(bad)}", flush=True)
    if stress:
        r = stress[0]
        print(f"  stress run: undecided subproblems {r.get('unknown')}, fallback to CP-F {r.get('fallback')}, "
              f"status {r.get('status')}", flush=True)
    print("SMOKE TEST OK" if not fails else "SMOKE TEST FAILED: " + "; ".join(fails), flush=True)
    sys.exit(0 if not fails else 1)
