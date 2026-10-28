"""Time and energy trade-off by weighted-sum sweep (Sections 3.3.4 and 5.5, Table 5, Figure 5).

1. Ten runs minimizing C_time and ten minimizing C_energy (raw units) give pooled bounds
   [T_lo, T_hi] and [E_lo, E_hi] for min-max normalization.
2. For w = 0.1 .. 0.9, ten runs each minimize F_w (Eq. 17); the GA maximizes -F_w, which keeps
   the ranking correct even when a tour improves on a bound.
3. Non-dominance (Eq. 18) is checked over all 110 tours, on the forecast and on ground truth.
All runs use the forecast matrix at 17:00 on day 40 with the nearest-neighbour seed.
"""
import time

import numpy as np

from gtst_common import (DECISION_DAY, DECISION_HOUR, Network, disrupted_scenario, forecast_matrix, load_model,
                         nearest_neighbor, route_costs, run_ga, save_json, seeded_population)

SEEDS_PER_WEIGHT = 10
WEIGHTS = [round(w, 1) for w in np.linspace(0.0, 1.0, 11)]


def nondominated(points):
    """Boolean mask of points (T, E) not dominated by any other point."""
    keep = []
    for i, (t, e) in enumerate(points):
        keep.append(not any(t2 <= t and e2 <= e and (t2 < t or e2 < e)
                            for j, (t2, e2) in enumerate(points) if j != i))
    return keep


def main():
    start = time.time()
    net = Network()
    t0 = DECISION_DAY * 24 + DECISION_HOUR
    ratio, tt = disrupted_scenario(net)
    pred = forecast_matrix(net, load_model(), ratio, t0)
    truth = net.to_matrix(tt[t0])
    greedy = nearest_neighbor(net, pred)

    def optimize(score, seed):
        def fitness(route):
            _, t, e = route_costs(net, route, pred)
            return -score(t, e)
        return run_ga(fitness, net.n_nodes, seed, initial_population=seeded_population(greedy, seed, net.n_nodes))

    def record(route, w, seed):
        _, tp, ep = route_costs(net, route, pred)
        _, tg, eg = route_costs(net, route, truth)
        return {"w_time": w, "seed": seed, "T_pred": float(tp), "E_pred": float(ep),
                "T_gt": float(tg), "E_gt": float(eg), "route": [int(x) for x in route]}

    tours = []
    for s in range(SEEDS_PER_WEIGHT):
        tours.append(record(optimize(lambda t, e: t, 1000 + s), 1.0, 1000 + s))
        tours.append(record(optimize(lambda t, e: e, 2000 + s), 0.0, 2000 + s))
    T_lo, T_hi = min(r["T_pred"] for r in tours), max(r["T_pred"] for r in tours)
    E_lo, E_hi = min(r["E_pred"] for r in tours), max(r["E_pred"] for r in tours)
    print(f"bounds: T [{T_lo:.1f}, {T_hi:.1f}]  E [{E_lo:.1f}, {E_hi:.1f}]")

    for k, w in enumerate(WEIGHTS):
        if w in (0.0, 1.0):
            continue
        F_w = lambda t, e, w=w: w * (t - T_lo) / (T_hi - T_lo) + (1 - w) * (e - E_lo) / (E_hi - E_lo)
        for s in range(SEEDS_PER_WEIGHT):
            seed = 3000 + 100 * k + s
            tours.append(record(optimize(F_w, seed), w, seed))
        print(f"w = {w:.1f} done")

    for space, kt, ke in (("pred", "T_pred", "E_pred"), ("gt", "T_gt", "E_gt")):
        for r, nd in zip(tours, nondominated([(r[kt], r[ke]) for r in tours])):
            r[f"nondominated_{space}"] = bool(nd)
    per_weight = {}
    for w in WEIGHTS:
        rows = [r for r in tours if r["w_time"] == w]
        per_weight[str(w)] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
                              for k in ("T_pred", "E_pred", "T_gt", "E_gt")}
    T = np.array([r["T_pred"] for r in tours]); E = np.array([r["E_pred"] for r in tours])
    out = {"bounds_pred": {"T": [T_lo, T_hi], "E": [E_lo, E_hi]}, "n_tours": len(tours),
           "n_nondominated_pred": sum(r["nondominated_pred"] for r in tours),
           "n_nondominated_gt": sum(r["nondominated_gt"] for r in tours),
           "pearson_T_E_pred": float(np.corrcoef(T, E)[0, 1]),
           "per_weight_mean_sd": per_weight, "tours": tours, "runtime_sec": time.time() - start}
    save_json(out, "tradeoff_sweep.json")
    print(f"{out['n_nondominated_pred']} of {len(tours)} non-dominated on the forecast, "
          f"{out['n_nondominated_gt']} on ground truth; r(T, E) = {out['pearson_T_E_pred']:.3f}")


if __name__ == "__main__":
    main()
