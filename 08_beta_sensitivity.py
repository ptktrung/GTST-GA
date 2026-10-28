"""Sensitivity to the congestion penalty beta (Section 5.6, Table 6).

For beta in {1.5, 2.5, 4.0}: ten GTST-GA runs (seeds 0-9) with beta used in both the greedy seed
and the fitness, scored on ground truth with the same beta. Because the raw energy index scales
with beta, two comparable quantities are also reported for every run: the excess delay
X = sum(max(0, tau - tau_ff)) in minutes, and the energy index re-evaluated at beta = 2.5.
"""
import numpy as np

from gtst_common import (ALPHA, BETA, DECISION_DAY, DECISION_HOUR, Network, disrupted_scenario, forecast_matrix,
                         load_model, nearest_neighbor, route_costs, run_ga, save_json, seeded_population, sum_fitness)

BETAS = (1.5, 2.5, 4.0)
N_RUNS = 10


def main():
    net = Network()
    t0 = DECISION_DAY * 24 + DECISION_HOUR
    ratio, tt = disrupted_scenario(net)
    pred = forecast_matrix(net, load_model(), ratio, t0)
    truth = net.to_matrix(tt[t0])
    out = {}
    for beta in BETAS:
        greedy = nearest_neighbor(net, pred, beta=beta)
        fitness = sum_fitness(net, pred, beta=beta, as_float=True)
        runs = {"dist": [], "time": [], "energy": [], "excess_delay": [], "energy_at_ref_beta": []}
        for seed in range(N_RUNS):
            route = run_ga(fitness, net.n_nodes, seed, initial_population=seeded_population(greedy, seed, net.n_nodes))
            d, t, e = route_costs(net, route, truth, beta=beta, as_float=True)
            runs["dist"].append(d); runs["time"].append(t); runs["energy"].append(e)
            runs["excess_delay"].append((e - ALPHA * d) / beta)
            runs["energy_at_ref_beta"].append(route_costs(net, route, truth, beta=BETA, as_float=True)[2])
        summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in runs.items()}
        out[str(beta)] = {"mean_sd": summary, "runs": runs}
        print(f"beta {beta}: " + "  ".join(f"{k} {m:.1f}±{s:.1f}" for k, (m, s) in summary.items()))
    save_json(out, "beta_sensitivity.json")


if __name__ == "__main__":
    main()
