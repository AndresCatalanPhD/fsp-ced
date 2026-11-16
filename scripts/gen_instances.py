"""Generate the benchmark: (m, n) pairs x tightness tau x 3 seeds; each base instance is saved in its
expandable-deadline (ED) version; the common-deadline (CD) version is derived by Instance.common_variant().

python scripts/gen_instances.py instances                    # main benchmark (54) and the example
python scripts/gen_instances.py instances_large --large      # 200-300 targets (18)
python scripts/gen_instances.py instances_chakim --chakim    # design of Cha and Kim (2010) (54)
python scripts/gen_instances.py instances_short --short      # short-window instances (36), ED only
python scripts/gen_instances.py instances_dispersed --dispersed   # arbitrary battery sets (24), ED only
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fspced.instance import generate, generate_chakim, generate_short, generate_dispersed, optima_example

PAIRS = [(3, 10), (3, 20), (3, 40), (6, 20), (6, 40), (6, 80), (9, 40), (9, 80), (9, 120)]
PAIRS_LARGE = [(9, 200), (12, 200), (12, 300)]   # scalability set (Section 6.6)
CK_M, CK_N, CK_PCT = [3, 6, 9], [18, 26], [20, 50, 80]   # design of Cha and Kim (2010)
PAIRS_SHORT = [(3, 20), (6, 40), (9, 60), (12, 80)]   # short-window set (Section 7)
TAUS_SHORT = [0.05, 0.08, 0.12]
PAIRS_DISP = [(6, 40), (9, 60), (12, 80)]             # dispersed set: arbitrary battery sets (Section 7)
REGIMES_DISP = [(False, 0.30), (True, 0.08)]          # (short window?, tau): capped window and short window
SEEDS_DISP = [1, 2, 3, 4]
TAUS = [0.3, 0.5]
SEEDS = [1, 2, 3]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "instances"
    large = len(sys.argv) > 2 and sys.argv[2] == "--large"
    chakim = len(sys.argv) > 2 and sys.argv[2] == "--chakim"
    short = len(sys.argv) > 2 and sys.argv[2] == "--short"
    os.makedirs(out, exist_ok=True)
    k = 0
    if short:
        for m, n in PAIRS_SHORT:
            for tau in TAUS_SHORT:
                for r in SEEDS:
                    seed = 5000000 + 10000 * m + 100 * n + int(round(tau * 100)) * 10 + r
                    inst = generate_short(m, n, tau, seed, name=f"sw_m{m}_n{n}_t{int(round(tau * 100)):02d}_r{r}_ED")
                    inst.save(os.path.join(out, inst.name.replace("_ED", "") + ".json"))
                    k += 1
        print(k, "short-window instances written to", out)
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[2] == "--dispersed":
        for m, n in PAIRS_DISP:
            for short_w, tau in REGIMES_DISP:
                for r in SEEDS_DISP:
                    seed = 8000000 + 10000 * m + 100 * n + (50 if short_w else 0) + r
                    nm = f"dp_m{m}_n{n}_{'s' if short_w else 'c'}{int(round(tau * 100)):02d}_r{r}_ED"
                    inst = generate_dispersed(m, n, tau, seed, short=short_w, name=nm)
                    inst.save(os.path.join(out, inst.name.replace("_ED", "") + ".json"))
                    k += 1
        print(k, "dispersed instances written to", out)
        sys.exit(0)
    if chakim:
        for m in CK_M:
            for n in CK_N:
                for pct in CK_PCT:
                    for r in SEEDS:
                        seed = 7000000 + 10000 * m + 100 * n + pct + r
                        inst = generate_chakim(m, n, pct, seed, name=f"ck_m{m}_n{n}_p{pct}_r{r}_ED")
                        inst.save(os.path.join(out, inst.name.replace("_ED", "") + ".json"))
                        k += 1
        print(k, "Cha-Kim instances written to", out)
        sys.exit(0)
    for m, n in (PAIRS_LARGE if large else PAIRS):
        for tau in TAUS:
            for s in SEEDS:
                seed = 10000 * m + 100 * n + 10 * int(round(tau * 10)) + s
                inst = generate(m, n, tau, seed, name=f"m{m}_n{n}_t{int(tau*100)}_r{s}_ED")
                inst.save(os.path.join(out, inst.name.replace("_ED", "") + ".json"))
                k += 1
    if not large:
        optima_example().save(os.path.join(out, "optima2017.json"))
    print(k, "instances written to", out)
