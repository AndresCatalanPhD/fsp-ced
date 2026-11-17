# FSP-CED — Fire Scheduling Problem with Common and Expandable Deadlines

Code, instances and experiment scripts for the paper *"Fire scheduling with common and expandable deadlines: a flow shop
characterization, valid inequalities, and a logic-based Benders decomposition"*
(Canut-de-Bon, Catalán and Yuraszeck). Code version 3, revision 1.

## Layout
```
fspced/instance.py   instance model, generators of the five instance sets, dominance pruning of modes
fspced/check.py      independent schedule checker (used on every reported plan)
fspced/flowshop.py   flow-shop view of the expandable deadline: CP model with one second-stage machine and no
                     crediting literals (monolithic model CP-F and subproblem of the decomposition), deficit and
                     overrun of a set of engagements
fspced/lbbd.py       logic-based Benders decomposition: master problem with Johnson and ladder inequalities
                     (Gurobi / HiGHS / CP-SAT), flow-shop subproblem, deficit cuts; switches for the ablation
fspced/cpsat.py      CP model with pairwise crediting literals (CP-P)
fspced/milp.py       time-indexed (TI) and disjunctive (DJ) MILPs; Gurobi if licensed, else HiGHS
fspced/heur.py       SGS-ED decoder, MWSPT rule, GA, IG-ED, and the CP models started from IG-ED (IG+CP, IG+CPF)
fspced/single.py     single-battery case: Johnson order and exact dynamic program
scripts/             generators, runners (resumable, shardable), analysis, checks and example figure
instances/           54 coastal instances (JSON); CD variants are derived in code
instances_chakim/    54 instances following the published design of Cha and Kim (2010), adapted to the FSP-CED
instances_large/     18 larger instances (200-300 targets)
instances_short/     36 short-window instances (expandable deadline only)
instances_dispersed/ 24 instances with arbitrary battery sets, capped and short windows (expandable deadline only)
FSP_CED_Colab.ipynb  end-to-end notebook used for the paper's runs
results/             raw results of every run reported in the paper (one JSON record per run);
                     results/superseded/ holds the records of the earlier code version 2 (not used)
paper_results/       tables, macros and figures generated from results/ by scripts/analyze.py
```
## Quick start
```
pip install -r requirements.txt
python scripts/verify_theory.py --n 40         # structural results on random instances (prints THEORY CHECK OK)
python scripts/smoke_test.py                   # every runner on a few instances (prints SMOKE TEST OK)
python scripts/run_heur.py --tl 10 --seeds 0,1,2
python scripts/run_milp.py --tl 300            # TI on all instances, DJ up to n=20
python scripts/run_cp.py   --tl 300            # CP-P, CP-F, IG+CP and LBBD
python scripts/run_cp.py   --tl 300 --methods LBBD-T,LBBD-P,LBBD-B,LBBD-N,LBBD-BN,LBBD-0,LBBD-S --variants ED --out results/ablation.jsonl
python scripts/run_cp.py   --tl 300 --methods IG+CP,IG+CPF            # CP models started from the plan of IG-ED
python scripts/root_bounds.py --out results/root.jsonl                # bounds of the master problem (Table 5)
python scripts/sensitivity.py --out results/sensitivity.jsonl         # gain under other caps and extensions
python scripts/analyze.py  --res results --out paper_results
```
Result files are named by instance set: `cp`, `heur`, `milp`, `ablation`, `root` (coastal), and the same names with
the prefixes `ck_`, `large_`, `short_` and `disp_`; `sensitivity.jsonl` holds the runs of `sensitivity.py` and
`theory.jsonl` the counts of `verify_theory.py --out`.

## Methods (names used in the result files)
| Name | Method |
|---|---|
| `LBBD` | decomposition of the paper: Johnson and ladder inequalities, flow-shop subproblem, deficit cuts |
| `LBBD-T` | the subproblem does not receive the modes of the master problem (decomposition in the space of targets) |
| `LBBD-P` | pairwise CP model in the subproblem |
| `LBBD-B` | without Johnson and ladder inequalities (battery loads only) |
| `LBBD-N` | no-good cuts (every deficit and overrun replaced by 1; still valid) |
| `LBBD-BN` | without the inequalities and with no-good cuts |
| `LBBD-0` | conventional design: targets only, battery loads, pairwise subproblem, lifted no-good cuts |
| `LBBD-S` | classical no-good cuts, **not valid** with an expandable deadline (kept to measure the flaw) |
| `CPF` | flow-shop CP model with the caps, load bounds and ladder inequalities as redundant constraints (CP-F) |
| `CPF0`, `CPS` | CP-F without these constraints, and CP-P with them (2×2 comparison of Section 7.2) |
| `CP` | CP model with pairwise crediting literals (CP-P) |
| `IG+CP`, `IG+CPF` | IG-ED for five seconds, then CP-P or CP-F started from its plan (same start as LBBD) |
| `ROOT` | bounds of the master problem before any Benders cut: load bounds, + ladder, + Johnson inequalities |
| `SENS` | LBBD on the coastal instances with another cap or scaled extensions (variant `k<cap>_s<scale>`) |
| `TI`, `DJ`, `TI-LP`, `DJ-LP` | time-indexed and disjunctive MILPs and their linear relaxations |
| `MWSPT`, `GA`, `IG` | dispatching rule, genetic algorithm, iterated greedy (IG-ED) |

## Instance format (JSON)
`m`, `D0`, `Dbar`, `battery_xy`, `meta` and a list of `targets`, each with value `w`, extension `delta`,
class `kind` (HV major combatant, FA fast attack craft, LC landing craft), position `x`, `y` (km) and
`modes`: a list of `{"batteries": [...], "p": duration}`. The common-deadline variant (CD) of an
instance sets every `delta` to 0 and `Dbar = D0` (`Instance.common_variant()`).

## Reproducibility
Every runner appends one JSON record per run and skips the runs already stored with the current version of the
method (`scripts/common.py`), so an interrupted campaign continues where it stopped. All instances are regenerated
bit-for-bit by `scripts/gen_instances.py` (`--large`, `--chakim`, `--short`, `--dispersed`).

## Results of the paper
`results/` holds the raw records of the campaign reported in the paper (Google Colab, two cores, 300 s per run,
Gurobi for the MILPs and the master problem, CP-SAT of OR-Tools 9.15). To regenerate every table and number of the
paper from them:
```
python scripts/analyze.py --res results --out paper_results
```

## Development history
The decomposition was developed on the coastal, large and Cha-Kim instance sets. `results/superseded/` keeps the
records of the earlier code version 2 on those sets (LBBD and three ablation variants, 576 runs): that version of LBBD
proved 51 of 54 coastal and 7 of 18 large optima under an expandable deadline and handed over to the monolithic model
in 19 of its 252 runs. Version 3, reported in the paper, replaced its subproblem by the flow shop model and added the Johnson
and ladder inequalities. These records are not used by `scripts/analyze.py`, which reads only the files directly in
`results/`. The short-window set was generated while version 3 was being written, and the dispersed set after the method was fixed.

## Citation
Please cite the paper and this archive (see `CITATION.cff`; the DOI is given by Zenodo).

## License
Code: MIT (see `LICENSE`). Instances and results: CC BY 4.0.
