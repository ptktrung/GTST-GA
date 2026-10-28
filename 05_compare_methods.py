"""N = 30 comparison of routing methods under the disruption scenario (Sections 4.3, 5.3, 5.4).

    python 05_compare_methods.py                 # all methods, then the summary table
    python 05_compare_methods.py GTST-GA Oracle-GA   # selected methods only (e.g. to run in parallel)
    python 05_compare_methods.py summary         # summary from saved per-method results

Every tour is scored on the ground-truth matrix at 17:00 on day 40, whatever matrix the
method optimized. Per-run results are saved to results/comparison/<method>.json; seed k of
every GA method is run k (k = 0..29).
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

from gtst_common import (DECISION_DAY, DECISION_HOUR, RESULTS_DIR, Network, disrupted_scenario, forecast_matrix,
                         load_model, nearest_neighbor, route_costs, run_ga, save_json, seeded_population, sum_fitness)

N_RUNS = 30
OUT_DIR = os.path.join(RESULTS_DIR, "comparison")

# name: (cost matrix used by the optimizer, nearest-neighbour seed in the initial population, mutation)
GA_METHODS = {
    "Static-GA": ("free_flow", False, "inversion"),
    "GTST-GA": ("forecast", True, "inversion"),
    "Oracle-GA": ("truth", True, "inversion"),
    "Ablation-NoMemetic": ("forecast", False, "inversion"),
    "Ablation-SwapMutation": ("forecast", True, "swap"),
}
ALL_METHODS = list(GA_METHODS) + ["2opt-Oropt", "Predictive-Greedy"]


def setup():
    net = Network()
    t0 = DECISION_DAY * 24 + DECISION_HOUR
    ratio, tt = disrupted_scenario(net)
    mats = {"forecast": forecast_matrix(net, load_model(), ratio, t0),
            "truth": net.to_matrix(tt[t0]),
            "free_flow": net.free_flow}
    return net, mats


def two_opt_or_opt(net, seed):
    """Distance-minimizing local search from a random tour: improving 2-opt reversals and Or-opt
    relocations of 1-3 consecutive nodes, repeated until no move improves the tour (Section 4.3)."""
    rs = np.random.RandomState(seed)
    route = list(rs.permutation(range(1, net.n_nodes)))

    def length(r):
        full = [0] + r + [0]
        return sum(net.dist[full[k], full[k + 1]] for k in range(len(full) - 1))

    best = length(route)
    improved = True
    while improved:
        improved = False
        for a in range(len(route) - 1):                       # 2-opt
            for b in range(a + 1, len(route)):
                cand = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                c = length(cand)
                if c < best - 1e-9:
                    route, best, improved = cand, c, True
        for seg in (1, 2, 3):                                 # Or-opt
            for a in range(len(route) - seg + 1):
                segment, rest = route[a:a + seg], route[:a] + route[a + seg:]
                for pos in range(len(rest) + 1):
                    cand = rest[:pos] + segment + rest[pos:]
                    c = length(cand)
                    if c < best - 1e-9:
                        route, best, improved = cand, c, True
    return np.array(route)


def run_method(name, net, mats):
    truth = mats["truth"]
    greedy = nearest_neighbor(net, mats["forecast"])
    res = {"dist": [], "time": [], "energy": [], "exec_sec": [], "routes": []}
    if name == "Predictive-Greedy":
        tours = [(greedy, 0.0)]
    elif name == "2opt-Oropt":
        tours = []
        for seed in range(N_RUNS):
            t = time.time()
            tours.append((two_opt_or_opt(net, seed), time.time() - t))
    else:
        matrix, memetic, mutation = GA_METHODS[name]
        fitness = sum_fitness(net, mats[matrix])
        tours = []
        for seed in range(N_RUNS):
            init = seeded_population(greedy, seed, net.n_nodes) if memetic else None
            t = time.time()
            route = run_ga(fitness, net.n_nodes, seed, initial_population=init, mutation_type=mutation)
            tours.append((route, time.time() - t))
    for route, secs in tours:
        d, t, e = route_costs(net, route, truth)
        res["dist"].append(float(d)); res["time"].append(float(t)); res["energy"].append(float(e))
        res["exec_sec"].append(secs); res["routes"].append([int(x) for x in route])
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump(res, f)
    print(f"{name:24s} time {np.mean(res['time']):7.1f} min  ({len(tours)} run(s))")


def summary():
    data = {m: json.load(open(os.path.join(OUT_DIR, f"{m}.json"))) for m in ALL_METHODS}
    ref = np.array(data["GTST-GA"]["time"])
    table = {}
    for m, r in data.items():
        row = {"n": len(r["time"])}
        for k in ("time", "energy", "dist"):
            row[f"{k}_mean"] = round(float(np.mean(r[k])), 2)
            row[f"{k}_sd"] = round(float(np.std(r[k])), 2)             # population SD, as in Table 4
        if m != "GTST-GA" and len(r["time"]) == len(ref):
            row["wilcoxon_p_vs_GTST-GA"] = float(stats.wilcoxon(ref, r["time"]).pvalue)
        table[m] = row
        print(f"{m:24s} {row['time_mean']:7.1f} ± {row['time_sd']:5.1f}   {row['energy_mean']:7.1f} ± {row['energy_sd']:5.1f}"
              f"   {row['dist_mean']:6.1f}   p={row.get('wilcoxon_p_vs_GTST-GA', float('nan')):.4f}")
    save_json(table, "comparison_summary.json")


if __name__ == "__main__":
    args = sys.argv[1:] or ALL_METHODS + ["summary"]
    todo = [a for a in args if a != "summary"]
    if todo:
        net, mats = setup()
        for name in todo:
            run_method(name, net, mats)
    if "summary" in args:
        summary()
