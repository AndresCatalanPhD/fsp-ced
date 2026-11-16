"""Aggregate the JSONL results into the LaTeX tables, figure and macros used by the paper.

python scripts/analyze.py --res results --out paper_results

Instance sets (folder of instances, prefix of the result files):
  Co  coastal benchmark        instances/          cp, heur, milp, ablation
  CK  design of Cha and Kim    instances_chakim/   ck_cp, ck_heur, ck_milp, ck_ablation
  La  large instances          instances_large/    large_cp, large_heur, large_ablation
  Sh  short-window instances   instances_short/    short_cp, short_heur, short_milp, short_ablation   (ED only)

Writes, in --out:
  numbers.tex            \\newcommand macros with every number quoted in the text
  tab_sets.tex           the four instance sets                                   (main text)
  tab_exact.tex          exact methods, one row per set and variant                (main text)
  tab_ablation.tex       ablation of the decomposition on the ED instances         (main text)
  tab_coastal.tex, tab_large.tex, tab_chakim.tex, tab_short.tex     per-size detail (supplement)
  tab_formulations.tex   DJ, TI and CP on the small instances                      (supplement)
  tab_heuristics.tex     MWSPT, GA and IG-ED                                       (supplement)
  tab_value.tex          value and structure of the expandable deadline            (supplement)
  Figure_3.pdf           instances solved to proven optimality over time
  summary.csv            one row per instance, variant and method
"""
import argparse, glob, json, math, os
from collections import defaultdict
from common import ROOT, version_of  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from fspced.instance import Instance

SETS = [("Co", "Coastal", "inst", "", ("CD", "ED")),
        ("CK", "Cha--Kim design", "inst_ck", "ck_", ("CD", "ED")),
        ("La", "Large", "inst_large", "large_", ("CD", "ED")),
        ("Sh", "Short-window", "inst_short", "short_", ("ED",)),
        ("Dp", "Dispersed", "inst_disp", "disp_", ("ED",))]
EXACT = [("TI", "TI"), ("CP", "CPP"), ("CPF", "CPF"), ("LBBD", "LB")]           # method, macro tag
ABL = [("LBBD", "LB", "LBBD"),
       ("LBBD-T", "LBT", "T: targets only"),
       ("LBBD-P", "LBP", "P: pairwise model"),
       ("LBBD-B", "LBB", "B: load bounds only"),
       ("LBBD-N", "LBN", "N: no-good cuts"),
       ("LBBD-BN", "LBBN", "B and N"),
       ("LBBD-0", "LBZ", "Conventional"),
       ("LBBD-S", "LBS", "Classical cuts (not valid)")]
WARM = [("CP", "CPP", "CP-P"), ("IG+CP", "WCPP", "CP-P, warm"), ("CPF", "CPF", "CP-F"),
        ("IG+CPF", "WCPF", "CP-F, warm"), ("LBBD", "LB", "LBBD")]
BOUND_OK = {"TI", "DJ", "CP", "CPF", "CPS", "CPF0", "IG+CP", "IG+CPF", "LBBD", "LBBD-B", "LBBD-P", "LBBD-N", "LBBD-0", "LBBD-T", "LBBD-BN"}
LP = {"TI-LP", "DJ-LP"}
NAN = float("nan")


def load(path):
    """Load path and every shard file next to it (e.g. cp.jsonl, cp_0of3.jsonl).  Records written by an older
    version of a method are ignored, as in the runners; among the others, the last record of every instance,
    variant, method and seed wins."""
    best = {}
    for f in sorted(set(glob.glob(path[:-6] + "*.jsonl"))):
        for l in open(f):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:            # line cut by a disconnection
                continue
            if "instance" not in r or "method" not in r:
                continue
            if str(r.get("code_version")) != version_of(r["method"]):
                continue                            # record of an older version of the method
            best[(r["instance"], r.get("variant"), r["method"], r.get("seed", 0))] = r
    return list(best.values())


def fmt(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:,.{d}f}".replace(",", "{,}")


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else NAN


class Macros:
    def __init__(self):
        self.d = {}

    def add(self, name, val):
        self.d[name] = val

    def text(self):
        return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in self.d.items())


class Data:
    """Records of one instance set, with best known values (BKV) and best valid bounds (UB)."""

    def __init__(self, a, key, label, inst_arg, prefix, variants):
        self.key, self.label, self.variants = key, label, variants
        self.meta = {}
        idir = getattr(a, inst_arg)
        for f in sorted(glob.glob(os.path.join(idir, "*.json"))):
            name = os.path.basename(f)[:-5]
            if name.startswith("optima"):
                continue
            I = Instance.load(f)
            self.meta[name] = {"m": I.m, "n": I.n, "tau": I.meta.get("tau"), "D0": I.D0, "Dbar": I.Dbar,
                               "modes": I.n_modes() / I.n, "givers": sum(t.delta > 0 for t in I.targets),
                               "pct": I.meta.get("single_pct"), "inst": I,
                               "win": "short" if I.meta.get("short") else "capped"}
        g = lambda nm: load(os.path.join(a.res, prefix + nm + ".jsonl"))
        self.cp, self.he, self.mi, self.ab = g("cp"), g("heur"), g("milp"), g("ablation")
        self.root = [r for r in g("root") if r["method"] == "ROOT"]
        for rs in (self.cp, self.he, self.mi, self.ab, self.root):
            rs[:] = [r for r in rs if r["instance"] in self.meta]
        for r in self.cp + self.ab:                # integer objective: value = floor(bound) is optimal
            if r.get("bound") is not None and r.get("value") is not None \
                    and r["value"] >= math.floor(r["bound"] + 1e-6):
                r["optimal"] = True
        for r in self.mi:
            r["optimal"] = r.get("status") == "Optimal" and r["method"] not in LP
        self.BKV, self.UB, self.best_sol = defaultdict(int), defaultdict(lambda: math.inf), {}
        for r in self.cp + self.he + self.mi + self.ab:
            k = (r["instance"], r["variant"])
            if r["method"] not in LP and r.get("value") is not None and r.get("check", True):
                if r["value"] > self.BKV[k]:
                    self.BKV[k] = r["value"]
                    if r.get("sol"):
                        self.best_sol[k] = r["sol"]
            b = r.get("value") if r["method"] in LP else (r.get("bound") if r["method"] in BOUND_OK else None)
            if b is not None and r.get("status") not in ("Infeasible",):
                self.UB[k] = min(self.UB[k], math.floor(b + 1e-6))
        for k in list(self.UB):
            self.UB[k] = max(self.UB[k], self.BKV[k])
        self.bad = [r for r in self.cp + self.he + self.mi + self.ab if r.get("check") is False]

    def gap(self, r):
        k = (r["instance"], r["variant"])
        ub = self.UB[k]
        return 100.0 * (ub - (r.get("value") or 0)) / ub if 0 < ub < math.inf else NAN

    def rpd(self, r):
        k = (r["instance"], r["variant"])
        return 100.0 * (self.BKV[k] - r["value"]) / self.BKV[k] if self.BKV[k] > 0 and r.get("value") is not None else NAN

    def bgap(self, r):
        """Distance of the bound of a run from the best known value (%): the dual side of its gap."""
        k = (r["instance"], r["variant"])
        if self.BKV[k] <= 0 or r.get("bound") is None:
            return NAN
        return 100.0 * max(0.0, math.floor(r["bound"] + 1e-6) - self.BKV[k]) / self.BKV[k]

    def runs(self, method, variant, names=None, pool=None):
        pool = pool if pool is not None else (self.mi if method in ("TI", "DJ") else self.cp + self.ab)
        return [r for r in pool if r["method"] == method and r["variant"] == variant
                and (names is None or r["instance"] in names)]

    def cell(self, method, variant, names=None, with_gap=True):
        sub = self.runs(method, variant, names)
        if not sub:
            return ["--", "--", "--"][: 3 if with_gap else 2], (0, 0, [], [])
        no = sum(1 for r in sub if r.get("optimal"))
        ts = [r["time"] for r in sub]
        gs = [self.gap(r) for r in sub]
        out = [f"{no}/{len(sub)}", fmt(mean(ts))]
        if with_gap:
            out.append(fmt(mean(gs), 2))
        return out, (no, len(sub), ts, gs)


def exact_macros(M, D):
    for v in D.variants:
        for meth, tag in EXACT + [("CPS", "CPS"), ("CPF0", "CPFZ"), ("IG+CP", "WCPP"), ("IG+CPF", "WCPF")]:
            _, (no, n, ts, gs) = D.cell(meth, v)
            sub = D.runs(meth, v)
            M.add(f"Dev{tag}{D.key}{v}", fmt(mean([D.rpd(r) for r in sub]), 2) if n else "--")      # primal side
            M.add(f"Bnd{tag}{D.key}{v}", fmt(mean([D.bgap(r) for r in sub]), 2) if n else "--")     # dual side
            M.add(f"Opt{tag}{D.key}{v}", no if n else "--")
            M.add(f"N{tag}{D.key}{v}", n if n else "--")
            M.add(f"Time{tag}{D.key}{v}", fmt(mean(ts)) if n else "--")
            M.add(f"MedTime{tag}{D.key}{v}", fmt(float(np.median(ts))) if n else "--")
            M.add(f"MaxTime{tag}{D.key}{v}", fmt(max(ts)) if n else "--")
            M.add(f"Gap{tag}{D.key}{v}", fmt(mean(gs), 2) if n else "--")
            M.add(f"MaxGap{tag}{D.key}{v}", fmt(np.nanmax(gs), 2) if n and not all(np.isnan(gs)) else "--")
        lb = D.runs("LBBD", v)
        tg = f"{D.key}{v}"
        M.add(f"LBfirst{tg}", sum(1 for r in lb if r.get("optimal") and (r.get("subproblems") or 0) <= 1
                                  and (r.get("cuts") or 0) == (r.get("johnson_cuts") or 0)) if lb else "--")
        M.add(f"LBmeanIt{tg}", fmt(mean([r.get("iters") or 0 for r in lb])) if lb else "--")
        M.add(f"LBmaxIt{tg}", max([r.get("iters") or 0 for r in lb]) if lb else "--")
        M.add(f"LBjohn{tg}", sum(r.get("johnson_cuts") or 0 for r in lb) if lb else "--")
        M.add(f"LBjfail{tg}", sum(1 for r in lb if (r.get("johnson_fails") or 0) > 0) if lb else "--")
        M.add(f"LBover{tg}", sum(r.get("overrun_cuts") or 0 for r in lb) if lb else "--")
        M.add(f"LBdef{tg}", sum(r.get("deficit_cuts") or 0 for r in lb) if lb else "--")
        nc = sum((r.get("overrun_cuts") or 0) + (r.get("deficit_cuts") or 0) for r in lb)
        M.add(f"LBtheta{tg}", fmt(sum(r.get("theta_sum") or 0 for r in lb) / nc) if nc else "--")
        M.add(f"LBunk{tg}", sum(1 for r in lb if (r.get("unknown") or 0) > 0) if lb else "--")
        M.add(f"LBfb{tg}", sum(1 for r in lb if (r.get("fallback") or 0) > 0) if lb else "--")
        M.add(f"LBdec{tg}", sum(r.get("decoder_hits") or 0 for r in lb) if lb else "--")
        M.add(f"LBfix{tg}", sum(r.get("fixed_hits") or 0 for r in lb) if lb else "--")
        M.add(f"LBfree{tg}", sum(r.get("free_hits") or 0 for r in lb) if lb else "--")
        M.add(f"LBsub{tg}", sum(r.get("subproblems") or 0 for r in lb) if lb else "--")
        for meth in ("MWSPT", "GA", "IG"):
            dd = [D.rpd(r) for r in D.he if r["method"] == meth and r["variant"] == v]
            M.add(f"Rpd{meth}{tg}", fmt(mean(dd), 2) if dd else "--")
            M.add(f"MaxRpd{meth}{tg}", fmt(np.nanmax(dd), 2) if dd and not all(np.isnan(dd)) else "--")
        tb = [r.get("t_best") for r in D.he if r["method"] == "IG" and r["variant"] == v]
        M.add(f"TbestIG{tg}", fmt(mean(tb), 2) if tb else "--")
        keys = [(x, v) for x in D.meta]
        M.add(f"Proven{tg}", sum(1 for k in keys if D.BKV[k] == D.UB[k] and D.UB[k] < math.inf))
    M.add(f"Num{D.key}", len(D.meta) if D.meta else "--")


def tab_sets(a, sets):
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lrrrrrrrr@{\\extracolsep\\fill}}", "\\toprule",
         "\\textbf{Set} & \\textbf{Inst.} & $m$ & $n$ & \\textbf{Modes} & \\textbf{Ext.\\ (\\%)} & $D_0$ & "
         "$\\bar D/D_0$ & \\textbf{Engaged (\\%)} \\\\", "\\midrule"]
    for D in sets:
        if not D.meta:
            continue
        mt = list(D.meta.values())
        rng = lambda k: (f"{min(x[k] for x in mt)}--{max(x[k] for x in mt)}" if min(x[k] for x in mt) != max(x[k] for x in mt)
                         else f"{mt[0][k]}")
        eng = [100.0 * len(D.best_sol[(x, "ED")]) / D.meta[x]["n"] for x in D.meta if (x, "ED") in D.best_sol]
        L.append(f"{D.label} & {len(mt)} & {rng('m')} & {rng('n')} & {mean([x['modes'] for x in mt]):.1f} & "
                 f"{100.0 * mean([x['givers'] / x['n'] for x in mt]):.0f} & {rng('D0')} & "
                 f"{mean([x['Dbar'] / x['D0'] for x in mt]):.1f} & {fmt(mean(eng), 0)} \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, "tab_sets.tex"), "w").write("\n".join(L) + "\n")


def tab_exact(a, sets, M):
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}ll rrr rrr rrr rrr@{\\extracolsep\\fill}}", "\\toprule",
         "& & \\multicolumn{3}{c}{\\textbf{TI}} & \\multicolumn{3}{c}{\\textbf{CP-P}} & "
         "\\multicolumn{3}{c}{\\textbf{CP-F}} & \\multicolumn{3}{c@{}}{\\textbf{LBBD}} \\\\",
         "\\cmidrule{3-5}\\cmidrule{6-8}\\cmidrule{9-11}\\cmidrule{12-14}",
         "\\textbf{Set} & & \\#Opt & Time & Gap & \\#Opt & Time & Gap & \\#Opt & Time & Gap & \\#Opt & Time & Gap \\\\",
         "\\midrule"]
    tot = defaultdict(lambda: [0, 0, [], []])
    for D in sets:
        for v in D.variants:
            cells = []
            for meth, tag in EXACT:
                c, (no, n, ts, gs) = D.cell(meth, v)
                cells += c
                for key in ((tag, v), (tag, "All")):
                    tot[key][0] += no; tot[key][1] += n; tot[key][2] += ts; tot[key][3] += gs
            L.append(f"{D.label} & {v} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    for v in ("CD", "ED", "All"):
        cells = []
        for meth, tag in EXACT:
            no, n, ts, gs = tot[(tag, v)]
            cells += [f"{no}/{n}" if n else "--", fmt(mean(ts)), fmt(mean(gs), 2)]
            M.add(f"Opt{tag}All{'' if v == 'All' else v}", no if n else "--")
            M.add(f"N{tag}All{'' if v == 'All' else v}", n if n else "--")
            M.add(f"Time{tag}All{'' if v == 'All' else v}", fmt(mean(ts)) if n else "--")
        L.append(f"All sets & {v if v != 'All' else ''} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, "tab_exact.tex"), "w").write("\n".join(L) + "\n")


def tab_ablation(a, sets, M):
    use = [D for D in sets if "ED" in D.variants and D.meta]
    head = " & ".join(f"\\multicolumn{{3}}{{c{'@{}' if q == len(use) - 1 else ''}}}"
                      f"{{\\textbf{{{D.label.replace(' design', '')}}}}}" for q, D in enumerate(use))
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}l" + " rrr" * len(use) + "@{\\extracolsep\\fill}}",
         "\\toprule", "& " + head + " \\\\",
         "".join(f"\\cmidrule{{{2 + 3 * q}-{4 + 3 * q}}}" for q in range(len(use))),
         "\\textbf{Variant} & " + " & ".join(["\\#Opt & Time & Iter."] * len(use)) + " \\\\", "\\midrule"]
    for meth, tag, label in ABL:
        cells = []
        for D in use:
            sub = D.runs(meth, "ED")
            no = sum(1 for r in sub if r.get("optimal"))
            wrong = sum(1 for r in sub if r.get("optimal") and r["value"] < D.BKV[(r["instance"], "ED")])
            cells += [f"{no}/{len(sub)}" if sub else "--", fmt(mean([r["time"] for r in sub])) if sub else "--",
                      fmt(mean([r.get("iters") or 0 for r in sub])) if sub else "--"]
            M.add(f"AblOpt{tag}{D.key}", no if sub else "--")
            M.add(f"AblN{tag}{D.key}", len(sub) if sub else "--")
            M.add(f"AblTime{tag}{D.key}", fmt(mean([r["time"] for r in sub])) if sub else "--")
            M.add(f"AblIt{tag}{D.key}", fmt(mean([r.get("iters") or 0 for r in sub])) if sub else "--")
            M.add(f"AblMaxIt{tag}{D.key}", max([r.get("iters") or 0 for r in sub]) if sub else "--")
            M.add(f"AblUnk{tag}{D.key}", sum(1 for r in sub if (r.get("unknown") or 0) > 0) if sub else "--")
            M.add(f"AblCuts{tag}{D.key}", sum((r.get("cuts") or 0) - (r.get("johnson_cuts") or 0) for r in sub)
                  if sub else "--")
            if meth == "LBBD-S":
                M.add(f"AblWrong{D.key}", wrong if sub else "--")
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, "tab_ablation.tex"), "w").write("\n".join(L) + "\n")
    for meth, tag, _ in ABL:                         # totals over the sets
        subs = [r for D in use for r in D.runs(meth, "ED")]
        M.add(f"AblOpt{tag}All", sum(1 for r in subs if r.get("optimal")) if subs else "--")
        M.add(f"AblN{tag}All", len(subs) if subs else "--")
    wr = [1 for D in use for r in D.runs("LBBD-S", "ED") if r.get("optimal") and r["value"] < D.BKV[(r["instance"], "ED")]]
    M.add("AblWrongAll", len(wr) if any(D.runs("LBBD-S", "ED") for D in use) else "--")


def group_table(a, D, fname, groups, group_of, head_cols, methods, heur=()):
    """Per-group detail of one set: methods = [(method, with_gap)], heur = heuristic methods (mean deviation)."""
    ncol = len(head_cols) + sum(3 if g else 2 for _, _, g in methods) + len(heur)
    spec = "l" * len(head_cols) + "".join(" rrr" if g else " rr" for _, _, g in methods) + (" " + "r" * len(heur) if heur else "")
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}" + spec + "@{\\extracolsep\\fill}}", "\\toprule"]
    h1, h2, c = ["" for _ in head_cols], list(head_cols), len(head_cols) + 1
    cm = ""
    for meth, lab, g in methods:
        k = 3 if g else 2
        h1.append(f"\\multicolumn{{{k}}}{{c}}{{\\textbf{{{lab}}}}}")
        h2 += ["\\#Opt", "Time"] + (["Gap"] if g else [])
        cm += f"\\cmidrule{{{c}-{c + k - 1}}}"
        c += k
    if heur:
        h1.append(f"\\multicolumn{{{len(heur)}}}{{c@{{}}}}{{\\textbf{{Deviation (\\%)}}}}")
        h2 += [("IG-ED" if h == "IG" else h) for h in heur]
        cm += f"\\cmidrule{{{c}-{c + len(heur) - 1}}}"
    L += [" & ".join(h1) + " \\\\", cm, " & ".join(h2) + " \\\\"]
    for v in D.variants:
        L += ["\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{'Common deadline (CD)' if v == 'CD' else 'Expandable deadline (ED)'}}}}} \\\\"]
        for gkey in groups:
            names = {x for x in D.meta if group_of(D.meta[x]) == gkey}
            if not names:
                continue
            cells = [str(x) for x in gkey]
            for meth, lab, g in methods:
                cells += D.cell(meth, v, names, with_gap=g)[0]
            for h in heur:
                cells.append(fmt(mean([D.rpd(r) for r in D.he if r["method"] == h and r["variant"] == v
                                       and r["instance"] in names]), 2))
            L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, fname), "w").write("\n".join(L) + "\n")


def coastal_extras(a, D, M):
    """Tables and macros specific to the coastal benchmark: formulations, heuristics, value of the extensions."""
    meta, BKV, best_sol = D.meta, D.BKV, D.best_sol
    GROUPS = sorted({(x["m"], x["n"]) for x in meta.values()})
    # ---- formulations on the small instances
    small = {x for x in meta if meta[x]["n"] <= 20}
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}l rrr rrr@{\\extracolsep\\fill}}", "\\toprule",
         "& \\multicolumn{3}{c}{\\textbf{Common deadline (CD)}} & \\multicolumn{3}{c@{}}{\\textbf{Expandable deadline (ED)}} \\\\",
         "\\cmidrule{2-4}\\cmidrule{5-7}",
         "\\textbf{Model} & \\textbf{\\#Opt} & \\textbf{Time} & \\textbf{LP gap} & \\textbf{\\#Opt} & \\textbf{Time} & \\textbf{LP gap} \\\\", "\\midrule"]
    for meth, lab, lp in (("DJ", "DJ (disjunctive MILP)", "DJ-LP"), ("TI", "TI (time-indexed MILP)", "TI-LP"),
                          ("CP", "CP-P (pairwise CP model)", None), ("CPF", "CP-F (flow-shop CP model)", None)):
        cells = []
        for v in ("CD", "ED"):
            sub = D.runs(meth, v, small)
            nopt = sum(1 for r in sub if r.get("optimal"))
            t = mean([r["time"] for r in sub])
            lpg = NAN
            if lp:
                lpg = mean([100.0 * (r["value"] - BKV[(r["instance"], v)]) / BKV[(r["instance"], v)]
                            for r in D.mi if r["method"] == lp and r["variant"] == v and r["instance"] in small
                            and r.get("value") is not None and BKV[(r["instance"], v)] > 0])
                M.add(f"LPgap{meth}{v}small", fmt(lpg))
            cells += [f"{nopt}/{len(sub)}" if sub else "--", fmt(t), fmt(lpg)]
            tag = {"CP": "CPP"}.get(meth, meth)
            M.add(f"Opt{tag}{v}small", nopt if sub else "--")
            M.add(f"Time{tag}{v}small", fmt(t))
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    M.add("NumSmall", len(small))
    open(os.path.join(a.out, "tab_formulations.tex"), "w").write("\n".join(L) + "\n")
    # ---- heuristics
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}rr rrrr rrrr@{\\extracolsep\\fill}}", "\\toprule",
         "& & \\multicolumn{4}{c}{\\textbf{Common deadline (CD)}} & \\multicolumn{4}{c@{}}{\\textbf{Expandable deadline (ED)}} \\\\",
         "\\cmidrule{3-6}\\cmidrule{7-10}",
         "$m$ & $n$ & MWSPT & GA & IG-ED & $t_{\\text{best}}$ & MWSPT & GA & IG-ED & $t_{\\text{best}}$ \\\\", "\\midrule"]
    for g in GROUPS + [None]:
        names = set(meta) if g is None else {x for x in meta if (meta[x]["m"], meta[x]["n"]) == g}
        cells = []
        for v in ("CD", "ED"):
            for meth in ("MWSPT", "GA", "IG"):
                cells.append(fmt(mean([D.rpd(r) for r in D.he if r["method"] == meth and r["variant"] == v
                                       and r["instance"] in names]), 2))
            cells.append(fmt(mean([r.get("t_best") for r in D.he if r["method"] == "IG" and r["variant"] == v
                                   and r["instance"] in names]), 2))
        if g is None:
            L.append("\\midrule")
        L.append((f"{g[0]} & {g[1]} & " if g else "\\multicolumn{2}{l}{All} & ") + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, "tab_heuristics.tex"), "w").write("\n".join(L) + "\n")
    for v in ("CD", "ED"):
        ig = defaultdict(list)
        for r in D.he:
            if r["method"] == "IG" and r["variant"] == v:
                ig[r["instance"]].append(D.rpd(r))
        M.add(f"IGhitsBKV{v}", sum(1 for x in ig.values() if min(x) <= 1e-9))
        M.add(f"IGn{v}", len(ig))
    M.add("NumSeeds", len({r.get("seed") for r in D.he if r["method"] == "IG"}))
    # ---- value of the expandable deadline
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}rr rrrrr@{\\extracolsep\\fill}}", "\\toprule",
         "$m$ & $n$ & Gain (\\%) & Ext.\\ engaged, CD (\\%) & Ext.\\ engaged, ED (\\%) & Rank ext. & Rank others \\\\", "\\midrule"]
    gains_all, rg_all, ro_all, gcd_all, ged_all = [], [], [], [], []
    pair = []                                       # per instance: (mean rank of extending targets, of the others)
    for (m, n) in GROUPS:
        names = [x for x in meta if (meta[x]["m"], meta[x]["n"]) == (m, n)]
        gains, gcd, ged, rg, ro = [], [], [], [], []
        for x in names:
            if BKV[(x, "CD")] > 0:
                gains.append(100.0 * (BKV[(x, "ED")] - BKV[(x, "CD")]) / BKV[(x, "CD")])
            I = meta[x]["inst"]
            for v, acc in (("CD", gcd), ("ED", ged)):
                s = best_sol.get((x, v))
                if s:
                    tot_g = sum(1 for t in I.targets if t.delta > 0)
                    if tot_g:
                        acc.append(100.0 * sum(1 for j in s if I.targets[int(j)].delta > 0) / tot_g)
            s = best_sol.get((x, "ED"))
            if s and len(s) > 1:
                comp = sorted((v_[1] + I.targets[int(j)].modes[v_[0]].p, int(j)) for j, v_ in s.items())
                rg1, ro1 = [], []
                for r_, (_, j) in enumerate(comp):
                    (rg1 if I.targets[j].delta > 0 else ro1).append(r_ / (len(comp) - 1))
                rg += rg1; ro += ro1
                if rg1 and ro1:
                    pair.append((float(np.mean(rg1)), float(np.mean(ro1))))
        gains_all += gains; rg_all += rg; ro_all += ro; gcd_all += gcd; ged_all += ged
        L.append(f"{m} & {n} & {fmt(mean(gains))} & {fmt(mean(gcd))} & {fmt(mean(ged))} & "
                 f"{fmt(mean(rg), 2)} & {fmt(mean(ro), 2)} \\\\")
    L += ["\\midrule", f"\\multicolumn{{2}}{{l}}{{All}} & {fmt(mean(gains_all))} & {fmt(mean(gcd_all))} & "
          f"{fmt(mean(ged_all))} & {fmt(mean(rg_all), 2)} & {fmt(mean(ro_all), 2)} \\\\", "\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, "tab_value.tex"), "w").write("\n".join(L) + "\n")
    M.add("GainMean", fmt(mean(gains_all))); M.add("GainMin", fmt(min(gains_all)) if gains_all else "--")
    M.add("GainMax", fmt(max(gains_all)) if gains_all else "--")
    M.add("GiversCD", fmt(mean(gcd_all), 0)); M.add("GiversED", fmt(mean(ged_all), 0))
    M.add("RankGivers", fmt(mean(rg_all), 2)); M.add("RankOthers", fmt(mean(ro_all), 2))
    # The targets of one plan are not independent observations, so the test is paired by instance: one difference
    # per instance between the mean rank of its extending targets and that of the others (Wilcoxon signed-rank).
    M.add("RankNinst", len(pair) if pair else "--")
    M.add("RankNearly", sum(1 for x, y in pair if x < y) if pair else "--")
    diffs = [x - y for x, y in pair]
    if len(diffs) >= 6 and any(abs(d_) > 1e-12 for d_ in diffs):
        u = stats.wilcoxon(diffs, alternative="less")
        M.add("RankP", "<0.001" if u.pvalue < 0.001 else f"={u.pvalue:.3f}")
    else:
        M.add("RankP", "--")


def tab_bysize(a, Sh, Dp):
    """Short-window and dispersed sets by size (ED): the four exact methods and the deviation of IG-ED."""
    meths = [("TI", "TI"), ("CP", "CP-P"), ("CPF", "CP-F"), ("LBBD", "LBBD")]
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lrr" + " rrr" * len(meths) + " r@{\\extracolsep\\fill}}",
         "\\toprule",
         "& & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{lab}}}}}" for _, lab in meths) + " & \\textbf{IG-ED} \\\\",
         "".join(f"\\cmidrule{{{4 + 3 * q}-{6 + 3 * q}}}" for q in range(len(meths))),
         "\\textbf{Set} & $m$ & $n$ & " + " & ".join(["\\#Opt & Time & Gap"] * len(meths)) + " & Dev. \\\\", "\\midrule"]
    blocks = [(Sh, "Short-window", None), (Dp, "Dispersed, capped", "capped"), (Dp, "Dispersed, short", "short")]
    first = True
    for D, label, win in blocks:
        groups = sorted({(x["m"], x["n"]) for x in D.meta.values() if win is None or x["win"] == win})
        if not groups:
            continue
        if not first:
            L.append("\\addlinespace")
        first = False
        for q, (m, n) in enumerate(groups):
            names = {x for x in D.meta if (D.meta[x]["m"], D.meta[x]["n"]) == (m, n)
                     and (win is None or D.meta[x]["win"] == win)}
            cells = [label if q == 0 else "", str(m), str(n)]
            for meth, _ in meths:
                cells += D.cell(meth, "ED", names, with_gap=True)[0]
            cells.append(fmt(mean([D.rpd(r) for r in D.he if r["method"] == "IG" and r["variant"] == "ED"
                                   and r["instance"] in names]), 2))
            L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, "tab_bysize.tex"), "w").write("\n".join(L) + "\n")


def tab_root(a, sets, M):
    """Strength of the master problem before any Benders cut (scripts/root_bounds.py): distance of the linear and
    integer bounds from the best known value, with the load bounds only, with the ladder inequalities, and with all
    Johnson inequalities; and number of instances on which the integer bound equals the best known value."""
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}l rrr rrr rrr@{\\extracolsep\\fill}}", "\\toprule",
         "& \\multicolumn{3}{c}{\\textbf{Linear bound (\\%)}} & \\multicolumn{3}{c}{\\textbf{Integer bound (\\%)}} & "
         "\\multicolumn{3}{c@{}}{\\textbf{Integer bound exact}} \\\\",
         "\\cmidrule{2-4}\\cmidrule{5-7}\\cmidrule{8-10}",
         "\\textbf{Set} & Loads & +Ladder & +Johnson & Loads & +Ladder & +Johnson & Loads & +Ladder & +Johnson \\\\",
         "\\midrule"]
    any_row = False
    parts = []                                      # the dispersed set is reported by window regime
    for D in sets:
        if "ED" not in D.variants or not D.meta:
            continue
        if D.key == "Dp":
            parts += [(D, "DpC", "Dispersed, capped", "capped"), (D, "DpS", "Dispersed, short", "short")]
        else:
            parts.append((D, D.key, D.label, None))
    for D, key, label, win in parts:
        recs = [r for r in D.root if D.BKV[(r["instance"], "ED")] > 0
                and (win is None or D.meta[r["instance"]]["win"] == win)]
        cells = []
        for kind in ("lp", "mip"):
            for t in "012":
                g = [100.0 * (r[kind + t] - D.BKV[(r["instance"], "ED")]) / D.BKV[(r["instance"], "ED")]
                     for r in recs if r.get(kind + t) is not None]
                gm = max(0.0, mean(g)) if g else NAN             # a bound cannot be below the best known value
                cells.append(fmt(gm, 2) if g else "--")
                M.add(f"Root{'LP' if kind == 'lp' else 'IP'}{'ZAJ'[int(t)]}{key}", fmt(gm, 2) if g else "--")
        for t in "012":
            ex = sum(1 for r in recs if r.get("mip" + t) is not None
                     and math.floor(r["mip" + t] + 1e-6) <= D.BKV[(r["instance"], "ED")])
            cells.append(f"{ex}/{len(recs)}" if recs else "--")
            M.add(f"RootEx{'ZAJ'[int(t)]}{key}", ex if recs else "--")
        M.add(f"RootN{key}", len(recs) if recs else "--")
        any_row = any_row or bool(recs)
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, "tab_root.tex"), "w").write("\n".join(L) + "\n")
    return any_row


def tab_warm(a, sets, M):
    """Monolithic CP models with and without the plan of IG-ED as a starting solution, ED instances: proven optima
    and mean deviation of the final plan from the best known value (%)."""
    use = [D for D in sets if "ED" in D.variants and D.meta]
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}l" + " rr" * len(WARM) + "@{\\extracolsep\\fill}}",
         "\\toprule",
         "& " + " & ".join(f"\\multicolumn{{2}}{{c{'@{}' if q == len(WARM) - 1 else ''}}}{{\\textbf{{{lab}}}}}"
                           for q, (_, _, lab) in enumerate(WARM)) + " \\\\",
         "".join(f"\\cmidrule{{{2 + 2 * q}-{3 + 2 * q}}}" for q in range(len(WARM))),
         "\\textbf{Set} & " + " & ".join(["\\#Opt & Dev."] * len(WARM)) + " \\\\", "\\midrule"]
    tot = defaultdict(lambda: [0, 0, []])
    for D in use:
        cells = []
        for meth, tag, _ in WARM:
            sub = D.runs(meth, "ED")
            no, dv = sum(1 for r in sub if r.get("optimal")), [D.rpd(r) for r in sub]
            cells += [f"{no}/{len(sub)}" if sub else "--", fmt(mean(dv), 2) if sub else "--"]
            tot[tag][0] += no; tot[tag][1] += len(sub); tot[tag][2] += dv
        L.append(f"{D.label} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    cells = []
    for meth, tag, _ in WARM:
        no, n, dv = tot[tag]
        cells += [f"{no}/{n}" if n else "--", fmt(mean(dv), 2) if n else "--"]
        if tag in ("WCPP", "WCPF"):
            M.add(f"Opt{tag}AllED", no if n else "--"); M.add(f"N{tag}AllED", n if n else "--")
    L.append("All sets & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    open(os.path.join(a.out, "tab_warm.tex"), "w").write("\n".join(L) + "\n")
    for tag, meth in (("WCPP", "IG+CP"), ("WCPF", "IG+CPF")):       # common deadline, all sets
        sub = [r for D in sets if "CD" in D.variants for r in D.runs(meth, "CD")]
        M.add(f"Opt{tag}AllCD", sum(1 for r in sub if r.get("optimal")) if sub else "--")
        M.add(f"N{tag}AllCD", len(sub) if sub else "--")
    lb = [r for D in sets for v in D.variants for r in D.runs("LBBD", v)]
    M.add("LBfbAll", sum(1 for r in lb if (r.get("fallback") or 0) > 0) if lb else "--")
    M.add("LBunkAll", sum(1 for r in lb if (r.get("unknown") or 0) > 0) if lb else "--")
    M.add("LBigOptAll", sum(1 for r in lb if r.get("optimal") and r.get("ig_value") == r.get("value")) if lb else "--")
    M.add("LBoptAll", sum(1 for r in lb if r.get("optimal")) if lb else "--")


def sens_macros(a, Co, M):
    """Gain of the expandable deadline over the common one under other caps and extensions (scripts/sensitivity.py)."""
    recs = [r for r in load(os.path.join(a.res, "sensitivity.jsonl")) if r.get("method") == "SENS"
            and r["instance"] in Co.meta and r.get("check", True)]
    names = {"k1.25_s1.0": "SensKlow", "k2.0_s1.0": "SensKhigh", "kinf_s1.0": "SensKinf",
             "k1.5_s0.5": "SensShalf", "k1.5_s2.0": "SensSdouble"}
    for v, nm in names.items():
        sub = [r for r in recs if r["variant"] == v and Co.BKV[(r["instance"], "CD")] > 0]
        g = [100.0 * (r["value"] - Co.BKV[(r["instance"], "CD")]) / Co.BKV[(r["instance"], "CD")] for r in sub]
        M.add(nm, fmt(mean(g)) if g else "--")
        M.add(nm + "Min", fmt(min(g)) if g else "--"); M.add(nm + "Max", fmt(max(g)) if g else "--")
        M.add(nm + "Opt", sum(1 for r in sub if r.get("optimal")) if sub else "--")
        M.add(nm + "N", len(sub) if sub else "--")
    allg = [r for r in recs if r["variant"] in names]
    M.add("SensOptAll", sum(1 for r in allg if r.get("optimal")) if allg else "--")
    M.add("SensNAll", len(allg) if allg else "--")
    return bool(recs)


def figure(a, sets):
    """Number of instances solved to proven optimality within a given time, all sets together."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"TI": ("TI (MILP)", "#8C8C8C", ":"), "CP": ("CP-P", "#DD8452", "--"), "CPF": ("CP-F", "#55A868", "-."),
             "LBBD-0": ("LBBD, conventional", "#8172B3", (0, (5, 1))), "LBBD": ("LBBD", "#C44E52", "-")}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.75), sharey=False)
    for ax, v in zip(axes, ("CD", "ED")):
        ntot = sum(len(D.meta) for D in sets if v in D.variants)
        tmax = 1.0
        for meth, (lab, col, ls) in style.items():
            ts = sorted(r["time"] for D in sets if v in D.variants for r in D.runs(meth, v) if r.get("optimal"))
            if not ts:
                continue
            tmax = max(tmax, ts[-1])
            ax.step([0.01] + ts + [ts[-1]], [0] + list(range(1, len(ts) + 1)) + [len(ts)], where="post", color=col,
                    ls=ls, lw=1.5 if meth == "LBBD" else 1.1, label=lab)
        ax.axhline(ntot, color="black", lw=0.5, ls=(0, (1, 3)))
        ax.set_xscale("log"); ax.set_xlim(0.05, max(400.0, tmax * 1.2)); ax.set_ylim(0, ntot * 1.05)
        ax.set_title(("Common deadline" if v == "CD" else "Expandable deadline") + f" ({ntot} instances)", fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8); ax.set_ylabel("proven optima", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(lw=0.3, alpha=0.6)
    axes[1].legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "Figure_3.pdf"), bbox_inches="tight")


def theory_macros(a, M):
    recs = [r for r in load(os.path.join(a.res, "theory.jsonl")) if r.get("method") == "THEORY"]
    names = {"A_n": "ThAn", "A_bad": "ThAbad", "B_n": "ThBn", "B_bad": "ThBbad", "B_exact_n": "ThBexact",
             "B_exact_bad": "ThBexactBad", "C_cert": "ThCert", "C_strict": "ThStrict", "C_bad": "ThCbad",
             "C_classical_n": "ThClassN", "C_classical_bad": "ThClassBad", "D_n": "ThDn", "D_bad": "ThDbad",
             "D_wrong_std": "ThWrongStd", "D_iters_deficit": "ThItDef", "D_iters_nogood": "ThItNG"}
    for k, nm in names.items():
        vals = [r[k] for r in recs if k in r]
        M.add(nm, sum(vals) if vals else "--")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="instances"); ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="paper_results"); ap.add_argument("--inst-large", default="instances_large")
    ap.add_argument("--inst-ck", default="instances_chakim"); ap.add_argument("--inst-short", default="instances_short")
    ap.add_argument("--inst-disp", default="instances_dispersed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = [Data(a, *s) for s in SETS]
    M = Macros()
    Co, CK, La, Sh, Dp = sets
    allrec = [r for D in sets for r in D.cp + D.he + D.mi + D.ab]
    tl = {}
    for r in allrec:
        if r.get("tl") is not None:
            tl.setdefault(r["method"], r["tl"])
    M.add("TLcp", fmt(tl.get("LBBD", tl.get("CP", NAN)), 0)); M.add("TLheur", fmt(tl.get("IG", NAN), 0))
    M.add("TLmilp", fmt(tl.get("TI", NAN), 0)); M.add("TLdj", fmt(tl.get("DJ", NAN), 0))
    backends = sorted({r.get("backend") for D in sets for r in D.mi if r.get("backend")})
    M.add("MILPsolver", "Gurobi" if backends == ["gurobi"] else ("HiGHS" if backends == ["highs"] else "/".join(backends) or "--"))
    masters = sorted({r.get("master") for D in sets for r in D.runs("LBBD", "ED") if r.get("master")})
    M.add("MasterSolver", {"gurobi": "Gurobi", "highs": "HiGHS", "cpsat": "CP-SAT"}.get(masters[0], masters[0]) if len(masters) == 1 else "/".join(masters) or "--")
    M.add("NumBadCheck", sum(len(D.bad) for D in sets))
    M.add("NumAll", sum(len(D.meta) for D in sets))
    M.add("NumSets", {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(sum(1 for D in sets if D.cp), "--"))
    M.add("NumAllCD", sum(len(D.meta) for D in sets if "CD" in D.variants))
    M.add("NumAllED", sum(len(D.meta) for D in sets if "ED" in D.variants))
    M.add("NumRuns", sum(len(D.meta) * len(D.variants) for D in sets))
    for D in sets:
        exact_macros(M, D)
    tab_sets(a, sets)
    tab_exact(a, sets, M)
    tab_ablation(a, sets, M)
    tab_root(a, sets, M)
    tab_warm(a, sets, M)
    tab_bysize(a, Sh, Dp)
    sens_macros(a, Co, M)
    ex = [("TI", "TI", True), ("CP", "CP-P", True), ("CPF", "CP-F", True), ("LBBD", "LBBD", True)]
    group_table(a, Co, "tab_coastal.tex", sorted({(x["m"], x["n"]) for x in Co.meta.values()}),
                lambda x: (x["m"], x["n"]), ["$m$", "$n$"], ex)
    group_table(a, La, "tab_large.tex", sorted({(x["m"], x["n"]) for x in La.meta.values()}),
                lambda x: (x["m"], x["n"]), ["$m$", "$n$"], ex[1:], heur=("MWSPT", "IG"))
    group_table(a, CK, "tab_chakim.tex", sorted({(x["m"], x["pct"]) for x in CK.meta.values()}),
                lambda x: (x["m"], x["pct"]), ["$m$", "Single (\\%)"],
                [("TI", "TI", False), ("CP", "CP-P", False), ("CPF", "CP-F", False), ("LBBD", "LBBD", True)],
                heur=("MWSPT", "IG"))
    group_table(a, Sh, "tab_short.tex", sorted({(x["m"], x["n"]) for x in Sh.meta.values()}),
                lambda x: (x["m"], x["n"]), ["$m$", "$n$"], ex, heur=("IG",))
    group_table(a, Dp, "tab_disp.tex", sorted({(x["m"], x["n"], x["win"]) for x in Dp.meta.values()}),
                lambda x: (x["m"], x["n"], x["win"]), ["$m$", "$n$", "Window"], ex, heur=("IG",))
    if Co.meta:
        coastal_extras(a, Co, M)
    if CK.meta:
        g = [100.0 * (CK.BKV[(x, "ED")] - CK.BKV[(x, "CD")]) / CK.BKV[(x, "CD")] for x in CK.meta
             if CK.BKV[(x, "CD")] > 0 and CK.BKV[(x, "ED")] > 0]
        M.add("GainCK", fmt(mean(g)))
    theory_macros(a, M)
    figure(a, sets)
    rows = []
    for D in sets:
        for r in D.cp + D.mi + D.ab + D.he:
            k = (r["instance"], r["variant"])
            rows.append({"set": D.key, "instance": r["instance"], "variant": r["variant"], "method": r["method"],
                         "seed": r.get("seed", 0), "m": D.meta[r["instance"]]["m"], "n": D.meta[r["instance"]]["n"],
                         "value": r.get("value"), "time": r.get("time"), "optimal": bool(r.get("optimal")),
                         "bound": r.get("bound"), "iters": r.get("iters"), "BKV": D.BKV[k],
                         "UB": D.UB[k] if D.UB[k] < math.inf else None})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    open(os.path.join(a.out, "numbers.tex"), "w").write(
        "% generated by scripts/analyze.py -- do not edit by hand\n" + M.text())
    print("checker failures:", sum(len(D.bad) for D in sets))
    print(M.text())


if __name__ == "__main__":
    main()
