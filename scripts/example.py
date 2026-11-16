"""Illustrative example (Section 3): the OPTIMA 2017 instance with and without expandable deadline.
Writes the Gantt chart of both optimal plans; the row W is the second stage of the flow shop (the window).
python scripts/example.py <out.pdf>"""
import sys, os
from common import ROOT
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from fspced.instance import Instance, optima_example
from fspced.cpsat import solve_cp
from fspced.heur import Decoder
from fspced.check import check


def example_instances():
    cd = optima_example()
    ed = Instance.from_dict(cd.to_dict())
    ed.name = "optima2017_ED"
    ed.targets[4].delta = 6      # T5: major combatant with counter-battery capability
    ed.targets[2].delta = 4      # T3: missile craft
    ed.Dbar = 50
    return cd, ed


def left_justify(inst, sol):
    dec = Decoder(inst, passes=1)
    order = sorted(sol, key=lambda j: (sol[j][1], j)) + [j for j in range(inst.n) if j not in sol]
    rules = np.full(inst.n, -1, dtype=np.int64)
    for j, (h, S) in sol.items():
        rules[j] = h
    return dec.solution(np.array(order, dtype=np.int64), rules)


def gantt(ax, inst, sol, title):
    cols = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for j, (h, S) in sol.items():
        md = inst.targets[j].modes[h]
        for i in md.batteries:
            ax.add_patch(Rectangle((S, i + 0.12), md.p, 0.76, facecolor=cols[j], edgecolor="black", lw=0.6))
            ax.text(S + md.p / 2, i + 0.5, f"T{j+1}", ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")
    # second stage of the flow shop (Proposition 2): the window is a machine that is idle for D0 time units and
    # processes the extension of every extending target after its engagement
    y0, hgt = -0.95, 0.62
    ax.add_patch(Rectangle((0, y0), inst.D0, hgt, facecolor="#EEEEEE", edgecolor="grey", lw=0.5, hatch="///"))
    ax.text(inst.D0 / 2, y0 + hgt / 2, r"idle ($D_0=41$)", ha="center", va="center", fontsize=7,
            bbox=dict(facecolor="#EEEEEE", edgecolor="none", pad=1.0))
    ax.axvline(inst.D0, color="black", ls="--", lw=1)
    if inst.Dbar > inst.D0:
        ax.axvline(inst.Dbar, color="grey", ls=":", lw=1.2)
        ax.text(inst.Dbar + 0.6, 2.86, r"$\bar D=50$", ha="left", va="center", fontsize=7.5, color="dimgrey")
        comp = sorted((S + inst.targets[j].modes[h].p, j) for j, (h, S) in sol.items())
        lim = inst.D0
        for C, j in comp:
            d = inst.targets[j].delta
            if d > 0:
                ax.add_patch(Rectangle((lim, y0), d, hgt, facecolor=cols[j], edgecolor="black", lw=0.6, alpha=0.85))
                ax.text(lim + d / 2, y0 + hgt / 2, f"+{d}", ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold")
                ax.annotate("", xy=(lim + 0.15, y0 + hgt), xytext=(C, 0.12),
                            arrowprops=dict(arrowstyle="->", lw=0.7, color=cols[j], ls="-"))
                lim += d
    val = sum(inst.targets[j].w for j in sol)
    ax.set_title(f"{title}: value {val}", fontsize=9)
    ax.set_xlim(0, 56); ax.set_ylim(-1.1, 3.05)
    ax.set_yticks([-0.64, 0.5, 1.5, 2.5]); ax.set_yticklabels(["W", "B1", "B2", "B3"], fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("time (min)", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Figure_2.pdf")
    cd, ed = example_instances()
    res = {}
    for inst in (cd, ed):
        r = solve_cp(inst, 30, workers=1)
        sol = left_justify(inst, r["sol"])
        ok, val, _ = check(inst, sol)
        assert ok and val == r["value"], (ok, val, r["value"])
        res[inst.name] = (r, sol)
        print(inst.name, r["status"], r["value"], sorted((j + 1, h, S, S + inst.targets[j].modes[h].p) for j, (h, S) in sol.items()))
    # CD-optimal plan evaluated under the ED rules (it stays feasible)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.75), sharey=True)
    gantt(axes[0], cd, res[cd.name][1], "(a) Common deadline")
    gantt(axes[1], ed, res[ed.name][1], "(b) Expandable deadline")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("written", out)
