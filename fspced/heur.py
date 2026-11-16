"""Heuristics for FSP-CED: serial schedule generation with expandable deadlines (SGS-ED),
the MWSPT dispatching rule of Canut-de-Bon et al. (2017), the permutation GA of the group's
earlier report, and the proposed iterated greedy (IG-ED)."""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from .instance import Instance


# ----------------------------------------------------------------------------- data
class Arrays:
    def __init__(self, inst: Instance):
        self.n, self.m, self.D0, self.Db = inst.n, inst.m, inst.D0, inst.Dbar
        self.w = np.array([t.w for t in inst.targets], dtype=np.int64)
        self.delta = np.array([t.delta for t in inst.targets], dtype=np.int64)
        K = inst.n_modes()
        qmax = max(len(md.batteries) for t in inst.targets for md in t.modes)
        self.mstart = np.zeros(self.n + 1, dtype=np.int64)
        self.mp = np.zeros(K, dtype=np.int64)
        self.marea = np.zeros(K, dtype=np.int64)
        self.mnb = np.zeros(K, dtype=np.int64)
        self.mbat = -np.ones((K, qmax), dtype=np.int64)
        k = 0
        for j, t in enumerate(inst.targets):
            self.mstart[j] = k
            for md in t.modes:
                self.mp[k] = md.p
                self.marea[k] = md.area
                self.mnb[k] = len(md.batteries)
                for q, i in enumerate(md.batteries):
                    self.mbat[k, q] = i
                k += 1
        self.mstart[self.n] = k
        self.minarea = np.array([inst.min_area(j) for j in range(self.n)], dtype=np.int64)
        self.minp = np.array([min(md.p for md in t.modes) for t in inst.targets], dtype=np.int64)


# ----------------------------------------------------------------------------- decoder
@njit(cache=True)
def _place_range(k_lo, k_hi, rule, busy, Ecum, D0, Db, mp, marea, mnb, mbat, free):
    best_k, best_S, best_C, best_a = -1, -1, 1 << 30, 1 << 30
    T = Db
    for k in range(k_lo, k_hi):
        p = mp[k]
        if p > T:
            continue
        # combined availability of the batteries of mode k
        for t in range(T):
            free[t] = 1
        for q in range(mnb[k]):
            i = mbat[k, q]
            for t in range(T):
                if busy[i, t]:
                    free[t] = 0
        run = 0
        for e in range(T):              # e = last period of the firing, C = e + 1
            if free[e]:
                run += 1
            else:
                run = 0
            if run >= p:
                C = e + 1
                lim = D0 + Ecum[C]
                if lim > Db:
                    lim = Db
                if C <= lim:
                    S = C - p
                    a = marea[k]
                    if rule == 0:
                        better = (C < best_C) or (C == best_C and a < best_a)
                    else:
                        better = (a < best_a) or (a == best_a and C < best_C)
                    if better:
                        best_k, best_S, best_C, best_a = k, S, C, a
                    break
    return best_k, best_S


@njit(cache=True)
def _place(j, rule, busy, Ecum, D0, Db, mstart, mp, marea, mnb, mbat, free):
    """Earliest feasible placement of target j.  rule = -1: mode of earliest completion;
    rule = -2: mode of smallest battery-time; rule = h >= 0: mode h (earliest completion if h does
    not fit).  Returns (mode, start) or (-1, -1)."""
    if rule >= 0:
        k0 = mstart[j] + rule
        k, S = _place_range(k0, k0 + 1, 0, busy, Ecum, D0, Db, mp, marea, mnb, mbat, free)
        if k >= 0:
            return k, S
        return _place_range(mstart[j], mstart[j + 1], 0, busy, Ecum, D0, Db, mp, marea, mnb, mbat, free)
    return _place_range(mstart[j], mstart[j + 1], -1 - rule, busy, Ecum, D0, Db, mp, marea, mnb,
                        mbat, free)


@njit(cache=True)
def decode(order, rules, n, m, D0, Db, w, delta, mstart, mp, marea, mnb, mbat, passes):
    """Serial SGS-ED.  Returns (value, sumC, mode[n], start[n]) with mode=-1 for rejected targets."""
    busy = np.zeros((m, Db), dtype=np.uint8)
    Ecum = np.zeros(Db + 2, dtype=np.int64)       # Ecum[t] = credit earned by completions < t
    free = np.zeros(Db, dtype=np.uint8)
    mode = -np.ones(n, dtype=np.int64)
    start = -np.ones(n, dtype=np.int64)
    value = 0
    sumC = 0
    for ps in range(passes):
        added = 0
        for idx in range(order.shape[0]):
            j = order[idx]
            if mode[j] >= 0:
                continue
            k, S = _place(j, rules[j], busy, Ecum, D0, Db, mstart, mp, marea, mnb, mbat, free)
            if k < 0:
                continue
            p = mp[k]
            C = S + p
            for q in range(mnb[k]):
                i = mbat[k, q]
                for t in range(S, C):
                    busy[i, t] = 1
            if delta[j] > 0:
                for t in range(C + 1, Db + 2):
                    Ecum[t] += delta[j]
            mode[j] = k - mstart[j]
            start[j] = S
            value += w[j]
            sumC += C
            added += 1
        if added == 0 or delta.max() == 0:
            break
    return value, sumC, mode, start


class Decoder:
    def __init__(self, inst: Instance, passes: int = 3):
        self.A = Arrays(inst)
        self.passes = passes
        self.calls = 0

    def __call__(self, order, rules):
        A = self.A
        self.calls += 1
        return decode(order, rules, A.n, A.m, A.D0, A.Db, A.w, A.delta, A.mstart, A.mp, A.marea,
                      A.mnb, A.mbat, self.passes)

    def solution(self, order, rules):
        v, sc, mode, start = self(order, rules)
        return {int(j): (int(mode[j]), int(start[j])) for j in range(self.A.n) if mode[j] >= 0}


# ----------------------------------------------------------------------------- MWSPT
def mwspt(inst: Instance) -> dict:
    """Dispatching rule of Canut-de-Bon et al. (2017): order targets by p_j W_j / w_j
    (smallest battery-time per unit of value first) and place each one at its earliest start
    in the mode of smallest battery-time; targets that do not fit are skipped."""
    t0 = time.time()
    dec = Decoder(inst, passes=1)
    A = dec.A
    order = np.array(sorted(range(A.n), key=lambda j: (A.minarea[j] / A.w[j], A.minp[j])), dtype=np.int64)
    rules = np.full(A.n, -2, dtype=np.int64)
    v, sc, mode, start = dec(order, rules)
    return {"value": int(v), "time": time.time() - t0, "sol": dec.solution(order, rules)}


# ----------------------------------------------------------------------------- GA (baseline)
def _ox(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = -np.ones(n, dtype=np.int64)
    child[a:b + 1] = p1[a:b + 1]
    used = set(p1[a:b + 1].tolist())
    fill = [g for g in np.concatenate([p2[b + 1:], p2[:b + 1]]) if g not in used]
    pos = list(range(b + 1, n)) + list(range(0, a))
    for q, g in zip(pos, fill):
        child[q] = g
    return child


def ga(inst: Instance, time_limit: float = 10.0, pop_size: int = 100, pmut: float = 0.5,
       seed: int = 0) -> dict:
    """Steady-state permutation GA (rank-based roulette, OX crossover, swap mutation and
    probabilistic replacement) as designed in the group's 2018 report; decoder = SGS-ED with the
    earliest-completion mode rule."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    dec = Decoder(inst, passes=1)
    n = inst.n
    rules = np.full(n, -1, dtype=np.int64)
    pop = [rng.permutation(n).astype(np.int64) for _ in range(pop_size)]
    fit = [float(dec(p, rules)[0]) for p in pop]
    best_i = int(np.argmax(fit))
    best, best_order, t_best = fit[best_i], pop[best_i].copy(), time.time() - t0
    it = 0
    while time.time() - t0 < time_limit:
        it += 1
        ranks = np.argsort(np.argsort(fit)) + 1.0
        pr = ranks / ranks.sum()
        i1, i2 = rng.choice(pop_size, 2, replace=False, p=pr)
        for c in (_ox(pop[i1], pop[i2], rng), _ox(pop[i2], pop[i1], rng)):
            if rng.random() < pmut:
                a, b = rng.choice(n, 2, replace=False)
                c[a], c[b] = c[b], c[a]
            fc = float(dec(c, rules)[0])
            r = int(rng.integers(pop_size))
            if fc > fit[r] or rng.random() < math.exp(-(fit[r] - fc)):
                pop[r], fit[r] = c, fc
            if fc > best:
                best, best_order, t_best = fc, c.copy(), time.time() - t0
    return {"value": int(best), "time": time.time() - t0, "t_best": t_best, "iters": it,
            "sol": dec.solution(best_order, rules), "evals": dec.calls}


# ----------------------------------------------------------------------------- IG-ED
def priority_orders(A: Arrays):
    """Initial priority lists.  (a) value per battery-minute; (b) 'credit first': targets whose
    extension pays for their own duration (delta_j >= p_j) first, by Johnson's rule, then the
    rest by value density; (c) MWSPT order."""
    n = A.n
    dens = A.w / A.minarea
    o1 = sorted(range(n), key=lambda j: -dens[j])
    gain = [j for j in range(n) if A.delta[j] >= A.minp[j] and A.delta[j] > 0]
    loss = [j for j in range(n) if A.delta[j] > 0 and A.delta[j] < A.minp[j]]
    rest = [j for j in range(n) if A.delta[j] == 0]
    o2 = (sorted(gain, key=lambda j: A.minp[j]) + sorted(loss, key=lambda j: -A.delta[j])
          + sorted(rest, key=lambda j: -dens[j]))
    o3 = sorted(range(n), key=lambda j: -(A.w[j] + A.delta[j] * dens.mean()) / A.minarea[j])
    return [np.array(o, dtype=np.int64) for o in (o1, o2, o3)]


def ig(inst: Instance, time_limit: float = 10.0, seed: int = 0, dmin: int = 2, dmax: int = 6,
       temp: float = 0.5, max_pos: int = 40, ls_tries: int = 8, r_max: int = 3) -> dict:
    """Iterated greedy with SGS-ED decoding.

    Solution = (priority list, mode rule per target).  Each iteration removes d targets from
    the list (half of the time among the engaged ones), reinserts them one by one at the best
    position (best value, then smallest sum of completion times), applies an insertion local
    search to rejected targets and accepts the new solution with the constant-temperature rule
    of Ruiz and Stuetzle (2007)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    dec = Decoder(inst, passes=3)
    A = dec.A
    n = A.n
    T_acc = temp * float(A.w.mean())

    def key(order, rules):
        v, sc, mode, _ = dec(order, rules)
        return (int(v), -int(sc)), mode

    cur, cur_rules, cur_key = None, None, None
    for o in priority_orders(A):
        for r in (-1, -2):
            rules = np.full(n, r, dtype=np.int64)
            k, _ = key(o, rules)
            if cur_key is None or k > cur_key:
                cur, cur_rules, cur_key = o.copy(), rules, k
    init_value = cur_key[0]
    best, best_rules, best_key, t_best = cur.copy(), cur_rules.copy(), cur_key, time.time() - t0

    nmodes = A.mstart[1:] - A.mstart[:-1]

    def best_insert(order_wo, rules, j, try_modes=True):
        """Best position (and mode gene) of target j in the list without j; rules[j] is updated."""
        L = len(order_wo)
        cand = range(L + 1) if L + 1 <= max_pos else np.unique(
            np.concatenate([[0, L], rng.choice(L + 1, max_pos - 2, replace=False)]))
        genes = [int(rules[j])]
        if try_modes:
            for g in (-1, -2, int(rng.integers(nmodes[j]))):
                if g not in genes:
                    genes.append(g)
        bk, bo, bg = None, None, genes[0]
        for g in genes:
            rules[j] = g
            for pos in cand:
                o = np.empty(L + 1, dtype=np.int64)
                o[:pos] = order_wo[:pos]
                o[pos] = j
                o[pos + 1:] = order_wo[pos:]
                k, _ = key(o, rules)
                if bk is None or k > bk:
                    bk, bo, bg = k, o, g
        rules[j] = bg
        return bk, bo

    dens = A.w / A.minarea
    it = 0
    while time.time() - t0 < time_limit:
        it += 1
        order, rules = cur.copy(), cur_rules.copy()
        _, mode = key(order, rules)
        engaged = np.flatnonzero(mode >= 0)
        rejected = np.flatnonzero(mode < 0)
        d = min(len(engaged), int(rng.integers(dmin, dmax + 1)))
        # ruin: random engaged targets, or a battery-time window (related removal)
        if d > 0 and rng.random() < 0.5:
            rem = list(rng.choice(engaged, d, replace=False))
        elif d > 0:
            _, _, md_, st_ = dec(order, rules)
            t_c = int(rng.integers(0, A.Db))
            dist = np.array([abs(st_[j] - t_c) for j in engaged])
            rem = list(engaged[np.argsort(dist)[:d]])
        else:
            rem = []
        # recreate: removed targets and a few rejected ones, by noisy value density
        extra = list(rng.choice(rejected, min(len(rejected), r_max), replace=False)) if len(rejected) else []
        pool = rem + extra
        mask = np.isin(order, pool)
        order = order[~mask]
        sc = np.array([dens[j] * rng.uniform(0.7, 1.3) + (A.delta[j] > 0) * rng.uniform(0.0, 0.5)
                       for j in pool])
        k = None
        for j in [pool[q] for q in np.argsort(-sc)]:
            k, order = best_insert(order, rules, j)
        if k is None:
            k, _ = key(order, rules)
        # local search: try to engage rejected targets
        _, mode = key(order, rules)
        rej = [j for j in range(n) if mode[j] < 0]
        rng.shuffle(rej)
        for j in rej[:ls_tries]:
            o_wo = order[order != j]
            g_old = rules[j]
            k2, o2 = best_insert(o_wo, rules, j, try_modes=False)
            if k2 > k:
                k, order = k2, o2
            else:
                rules[j] = g_old
        if k >= cur_key or rng.random() < math.exp(-(cur_key[0] - k[0] + 1) / T_acc if T_acc > 0 else -1e9):
            cur, cur_rules, cur_key = order, rules, k
        if k > best_key:
            best, best_rules, best_key, t_best = order.copy(), rules.copy(), k, time.time() - t0
    return {"value": int(best_key[0]), "time": time.time() - t0, "t_best": t_best, "iters": it,
            "init": int(init_value), "sol": dec.solution(best, best_rules), "evals": dec.calls}


# ----------------------------------------------------------------------------- hybrid
def ig_cp(inst: Instance, time_limit: float = 60.0, ig_time: float = 5.0, seed: int = 0,
          workers: int = 2) -> dict:
    """IG-ED for ig_time seconds, then CP-SAT warm-started with the IG plan for the rest."""
    from .cpsat import solve_cp
    t0 = time.time()
    g = ig(inst, ig_time, seed=seed)
    r = solve_cp(inst, max(1.0, time_limit - (time.time() - t0)), workers=workers, hint=g["sol"],
                 seed=seed)
    val, sol = g["value"], g["sol"]
    if r["value"] is not None and r["value"] > val:
        val, sol = r["value"], r["sol"]
    return {"value": int(val), "time": time.time() - t0, "bound": r["bound"], "optimal": r["optimal"],
            "ig_value": g["value"], "sol": sol, "status": r["status"]}


def ig_cpf(inst: Instance, time_limit: float = 60.0, ig_time: float = 5.0, seed: int = 0,
           workers: int = 2) -> dict:
    """IG-ED for ig_time seconds, then the flow-shop CP model (CP-F) warm-started with the IG plan."""
    from .flowshop import solve_cpf
    t0 = time.time()
    g = ig(inst, ig_time, seed=seed)
    r = solve_cpf(inst, max(1.0, time_limit - (time.time() - t0)), workers=workers, hint=g["sol"], seed=seed)
    val, sol = g["value"], g["sol"]
    if r["value"] is not None and r["value"] > val:
        val, sol = r["value"], r["sol"]
    return {"value": int(val), "time": time.time() - t0, "bound": r["bound"], "optimal": r["optimal"],
            "ig_value": g["value"], "sol": sol, "status": r["status"]}
