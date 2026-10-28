"""Two-cycle re-planning with warm-starting (Section 5.10, Table 10).

Cycle 1 (16:00, day 40): forecast, nearest-neighbour seed, 600 generations, seed 0.
Cycle 2 (17:00): new forecast and a reduced budget of 300 generations, run 30 times with
    cold start: nearest-neighbour seed on the new forecast + random tours (seeds 100-129)
    warm start: the cycle-1 tour + random tours (seeds 200-229)
All tours are scored on ground truth at 17:00. The cycle-1 tour is also scored at 17:00
without re-planning, as a reference. No part of the route is driven between the cycles.
"""
import time

import numpy as np
from scipy import stats

from gtst_common import (DECISION_DAY, Network, disrupted_scenario, forecast_matrix, load_model, nearest_neighbor,
                         route_costs, run_ga, save_json, seeded_population, sum_fitness)

N_RUNS = 30
GEN_FULL, GEN_REPLAN = 600, 300


def timed_forecast(net, model, ratio, t):
    start = time.time()
    m = forecast_matrix(net, model, ratio, t)
    return m, time.time() - start


def main():
    net = Network()
    model = load_model()
    t2 = DECISION_DAY * 24 + 17
    t1 = t2 - 1
    ratio, tt = disrupted_scenario(net, last_hour=t2)
    pred1, lat1 = timed_forecast(net, model, ratio, t1)
    pred2, lat2 = timed_forecast(net, model, ratio, t2)
    truth = net.to_matrix(tt[t2])

    start = time.time()
    plan1 = run_ga(sum_fitness(net, pred1, as_float=True), net.n_nodes, 0, generations=GEN_FULL,
                   initial_population=seeded_population(nearest_neighbor(net, pred1), 0, net.n_nodes))
    plan1_sec = time.time() - start

    fit2 = sum_fitness(net, pred2, as_float=True)
    greedy2 = nearest_neighbor(net, pred2)
    runs = {k: [] for k in ("cold_time", "cold_energy", "cold_sec", "warm_time", "warm_energy", "warm_sec")}
    for s in range(N_RUNS):
        for kind, seed_tour, seed in (("cold", greedy2, 100 + s), ("warm", plan1, 200 + s)):
            start = time.time()
            route = run_ga(fit2, net.n_nodes, seed, generations=GEN_REPLAN,
                           initial_population=seeded_population(seed_tour, seed, net.n_nodes))
            runs[f"{kind}_sec"].append(time.time() - start)
            _, t, e = route_costs(net, route, truth, as_float=True)
            runs[f"{kind}_time"].append(t)
            runs[f"{kind}_energy"].append(e)
        print(f"run {s:2d}: cold {runs['cold_time'][-1]:.1f}  warm {runs['warm_time'][-1]:.1f} min")

    _, keep_t, keep_e = route_costs(net, plan1, truth, as_float=True)
    w, c = np.array(runs["warm_time"]), np.array(runs["cold_time"])
    rng = np.random.default_rng(1)
    boot = [rng.choice(c, N_RUNS).mean() - rng.choice(w, N_RUNS).mean() for _ in range(10000)]
    out = {
        "forecast_latency_sec": {"cycle1": lat1, "cycle2": lat2},
        "cycle1_ga_sec": plan1_sec,
        "cycle1_plan_scored_at_17h": {"time": keep_t, "energy": keep_e},
        "cold": {k: [float(np.mean(runs[f"cold_{k}"])), float(np.std(runs[f"cold_{k}"]))] for k in ("time", "energy", "sec")},
        "warm": {k: [float(np.mean(runs[f"warm_{k}"])), float(np.std(runs[f"warm_{k}"]))] for k in ("time", "energy", "sec")},
        "cold_minus_warm_time": {"mean": float((c - w).mean()),
                                 "boot_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
        "wilcoxon_p": float(stats.wilcoxon(w, c).pvalue),
        "mannwhitney_p": float(stats.mannwhitneyu(w, c, alternative="two-sided").pvalue),
        "runs_better_than_keeping_plan": {"warm": int((w < keep_t).sum()), "cold": int((c < keep_t).sum())},
        "runs": runs, "cycle1_route": [int(x) for x in plan1],
    }
    save_json(out, "rolling_horizon.json")
    print(f"warm {out['warm']['time'][0]:.1f} ± {out['warm']['time'][1]:.1f}, cold {out['cold']['time'][0]:.1f} ± "
          f"{out['cold']['time'][1]:.1f}, keep plan {keep_t:.1f}; Wilcoxon p = {out['wilcoxon_p']:.4g}")


if __name__ == "__main__":
    main()
