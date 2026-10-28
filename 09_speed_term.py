"""Speed-dependent energy term, Eq. (14) (Section 5.9, Table 9).

Part 1: share of arcs whose ground-truth speed exceeds V_EFF at 03:00 and 17:00 on day 40
        (undisrupted data).
Part 2: the beta sensitivity of 08_beta_sensitivity.py repeated with the speed term
        (17:00, disrupted), recording the speed-term share of each tour's energy index.
Part 3: off-peak check at 03:00 on day 40 without disruption, beta = 2.5, with and without
        the speed term (gamma = 0.8 and gamma = 0).
"""
import numpy as np

from gtst_common import (ALPHA, DECISION_DAY, DECISION_HOUR, Network, disrupted_scenario, forecast_matrix,
                         load_model, run_ga, save_json, seeded_population, tour_arcs)

V_EFF = 28.0        # km/h
GAMMA = 0.8
N_RUNS = 10


def costs(net, route, time_mat, beta, gamma):
    """Distance, C_time and the extended energy index of Eq. (14); speed = 60 d / tau in km/h."""
    src, tgt = tour_arcs(route)
    d, t = net.dist[src, tgt], time_mat[src, tgt]
    congestion = np.clip(t - net.free_flow[src, tgt], 0, None)
    speed = np.where(t > 1e-6, d / (t / 60.0), 0.0)
    over = np.clip(speed - V_EFF, 0, None)
    e = ALPHA * d + beta * congestion + gamma * d * over ** 2 / 100.0
    speed_part = gamma * d * over ** 2 / 100.0
    return float(d.sum()), float(t.sum()), float(e.sum()), float(speed_part.sum())


def greedy_tour(net, time_mat, beta, gamma):
    unvisited, current, route = set(range(1, net.n_nodes)), 0, []
    while unvisited:
        best, best_cost = None, float("inf")
        for k in unvisited:
            tau, d = time_mat[current, k], net.dist[current, k]
            excess = max(0.0, tau - net.free_flow[current, k])
            speed = d / (tau / 60.0) if tau > 1e-6 else 0.0
            cost = tau + ALPHA * d + beta * excess + gamma * (max(0.0, speed - V_EFF) ** 2 * d / 100.0)
            if cost < best_cost:
                best, best_cost = k, cost
        route.append(best)
        unvisited.remove(best)
        current = best
    return np.array(route)


def experiment(net, pred, truth, beta, gamma):
    greedy = greedy_tour(net, pred, beta, gamma)

    def fitness(route):
        _, t, e, _ = costs(net, route, pred, beta, gamma)
        return 1.0 / (t + e + 1e-6)

    runs = {"dist": [], "time": [], "energy": [], "speed_term": [], "routes": []}
    for seed in range(N_RUNS):
        route = run_ga(fitness, net.n_nodes, seed, initial_population=seeded_population(greedy, seed, net.n_nodes))
        d, t, e, sp = costs(net, route, truth, beta, gamma)
        for k, v in zip(("dist", "time", "energy", "speed_term"), (d, t, e, sp)):
            runs[k].append(v)
        runs["routes"].append([int(x) for x in route])
    summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in runs.items() if k != "routes"}
    print(f"beta {beta}, gamma {gamma}: " + "  ".join(f"{k} {m:.1f}±{s:.1f}" for k, (m, s) in summary.items()))
    return {"mean_sd": summary, "runs": runs}


def main():
    net = Network()
    model = load_model()
    out = {"v_eff_kmh": V_EFF, "gamma": GAMMA}

    dvec = net.dist[net.edge_i, net.edge_j]
    for label, hour in (("03:00", 3), ("17:00", DECISION_HOUR)):
        speed = dvec / (net.travel_time[DECISION_DAY * 24 + hour] / 60.0)
        out[f"share_arcs_above_v_eff_{label}"] = float(np.mean(speed > V_EFF))
        print(f"{label}: {100 * out[f'share_arcs_above_v_eff_{label}']:.1f}% of arcs faster than {V_EFF} km/h")

    t0 = DECISION_DAY * 24 + DECISION_HOUR
    ratio, tt = disrupted_scenario(net)
    pred, truth = forecast_matrix(net, model, ratio, t0), net.to_matrix(tt[t0])
    out["peak_disrupted"] = {str(b): experiment(net, pred, truth, b, GAMMA) for b in (1.5, 2.5, 4.0)}

    t3 = DECISION_DAY * 24 + 3
    pred3, truth3 = forecast_matrix(net, model, net.ratio, t3), net.to_matrix(net.travel_time[t3])
    out["offpeak_undisrupted"] = {f"gamma_{g}": experiment(net, pred3, truth3, 2.5, g) for g in (GAMMA, 0.0)}
    save_json(out, "speed_term.json")


if __name__ == "__main__":
    main()
