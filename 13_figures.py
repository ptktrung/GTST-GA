"""Figures 1-6 and the representative-run table (Table 3), from the saved results.

Figures 2 and 3 use seed 0 of each GA method. The seed-0 runs are repeated here to record the
convergence history, and the tours are checked against run 0 in results/comparison/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gtst_common import (DECISION_DAY, DECISION_HOUR, DISRUPTED_NODES, FIGURES_DIR, RESULTS_DIR, Network,
                         disrupted_scenario, forecast_matrix, load_json, load_model, nearest_neighbor, route_costs,
                         run_ga, save_json, seeded_population, sum_fitness)


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig1_forecast():
    val = load_json("forecast_validation.json")
    hist = np.array(val["training_log"]["history_epoch_train_val"])
    pred, actual = np.load(os.path.join(RESULTS_DIR, "pred_vs_actual_test.npy")).T
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(hist[:, 0], hist[:, 1], label="training loss")
    ax[0].plot(hist[:, 0], hist[:, 2], label="validation loss")
    ax[0].set(xlabel="Epoch", ylabel="MSE of log travel-time ratio", title="GTST-Lite training")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    idx = np.random.default_rng(0).choice(len(pred), 5000, replace=False)
    ax[1].scatter(actual[idx], pred[idx], s=4, alpha=0.3)
    lim = [0, float(max(actual.max(), pred.max()))]
    ax[1].plot(lim, lim, "r--", linewidth=1, label="y = x")
    m = val["test"]
    ax[1].set(xlabel="Observed travel time (min)", ylabel="Forecast travel time (min)",
              title=f"Test split (5,000-point sample)\nMAE {m['MAE_min']:.2f} min, RMSE {m['RMSE_min']:.2f} min, MAPE {m['MAPE_pct']:.1f}%")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    save(fig, "figure1_forecast_validation.png")


def seed0_runs():
    """Seed-0 runs with convergence history; also writes Table 3."""
    net = Network()
    t0 = DECISION_DAY * 24 + DECISION_HOUR
    ratio, tt = disrupted_scenario(net)
    pred = forecast_matrix(net, load_model(), ratio, t0)
    truth = net.to_matrix(tt[t0])
    greedy = nearest_neighbor(net, pred)
    runs, table = {}, {}
    for name, mat, memetic in (("Static-GA", net.free_flow, False), ("GTST-GA", pred, True), ("Oracle-GA", truth, True)):
        hist = []
        route = run_ga(sum_fitness(net, mat, as_float=True), net.n_nodes, 0,
                       initial_population=seeded_population(greedy, 0, net.n_nodes) if memetic else None,
                       on_generation=lambda ga: hist.append(1.0 / ga.best_solution()[1]))
        saved = json.load(open(os.path.join(RESULTS_DIR, "comparison", f"{name}.json")))["routes"][0]
        if list(map(int, route)) != saved:
            print(f"warning: seed-0 tour of {name} differs from results/comparison")
        runs[name] = {"route": route, "history": hist}
    runs["Predictive-Greedy"] = {"route": greedy, "history": None}
    for name, r in runs.items():
        d, t, e = route_costs(net, r["route"], truth, as_float=True)
        table[name] = {"dist_km": d, "time_min": t, "energy_index": e, "route": [int(x) for x in r["route"]]}
    save_json(table, "representative_run_seed0.json")
    return net, runs, table


def fig2_routes(net, table):
    xy = net.nodes[["longitude", "latitude"]].values
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [("Static-GA", "A) Static-GA (traffic-blind)"), ("Predictive-Greedy", "B) Predictive-Greedy"),
              ("GTST-GA", "C) GTST-GA"), ("Oracle-GA", "D) Oracle-GA (perfect information)")]
    for ax, (name, title) in zip(axes.ravel(), panels):
        r = table[name]
        path = xy[[0] + r["route"] + [0]]
        ax.scatter(xy[:, 0], xy[:, 1], c="lightgray", s=30, zorder=2)
        ax.scatter(xy[list(DISRUPTED_NODES), 0], xy[list(DISRUPTED_NODES), 1], c="orange", s=140, alpha=0.6, zorder=3,
                   label="disrupted node")
        ax.scatter(xy[0, 0], xy[0, 1], c="red", marker="s", s=90, zorder=5, label="depot")
        ax.plot(path[:, 0], path[:, 1], c="tab:blue", linewidth=1.6, zorder=4)
        ax.text(0.98, 0.98, f"Distance {r['dist_km']:.1f} km\nTime {r['time_min']:.1f} min\n"
                            f"Energy index {r['energy_index']:.1f} kWh-eq.",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    save(fig, "figure2_routes_seed0.png")


def fig3_convergence(runs):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ("Static-GA", "GTST-GA", "Oracle-GA"):
        h = np.array(runs[name]["history"])
        ax.plot(np.arange(1, len(h) + 1), 100 * h / h[0], label=name)
    ax.set(xlabel="Generation", ylabel="Best objective F (% of generation 1)", title="GA convergence, seed 0")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "figure3_convergence.png")


def fig4_ablation():
    names = [("GTST-GA", "GTST-GA"), ("Ablation-NoMemetic", "No nearest-neighbour seed"),
             ("Ablation-SwapMutation", "Swap mutation")]
    data = [json.load(open(os.path.join(RESULTS_DIR, "comparison", f"{m}.json")))["time"] for m, _ in names]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.boxplot(data, tick_labels=[lab for _, lab in names])
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(d)), d, s=10, alpha=0.6)
    ax.set(ylabel="Travel time on ground truth (min)", title="GA operator ablation, N = 30 runs each")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "figure4_operator_ablation.png")


def fig5_tradeoff():
    res = load_json("tradeoff_sweep.json")
    tours = res["tours"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (kt, ke, kn, title) in zip(axes, [("T_pred", "E_pred", "nondominated_pred", "Costs on the forecast"),
                                            ("T_gt", "E_gt", "nondominated_gt", "Costs on ground truth")]):
        sc = ax.scatter([r[kt] for r in tours], [r[ke] for r in tours], c=[r["w_time"] for r in tours],
                        cmap="viridis", vmin=0, vmax=1, s=22, alpha=0.8)
        nd = [r for r in tours if r[kn]]
        ax.scatter([r[kt] for r in nd], [r[ke] for r in nd], facecolor="none", edgecolor="red", s=90, linewidth=1.5,
                   label=f"non-dominated ({len(nd)} of {len(tours)})")
        ax.set(xlabel="Travel time (min)", ylabel="Energy index (kWh-eq.)", title=title)
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=axes, shrink=0.85).set_label("time weight w")
    save(fig, "figure5_tradeoff.png")


def fig6_scalability():
    res = load_json("scalability.json")["sizes"]
    n = sorted(int(k) for k in res)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(n, [res[str(k)]["mean_sec"] for k in n], yerr=[res[str(k)]["sd_sec"] for k in n],
                marker="o", capsize=4)
    ax.set(xlabel="Number of nodes", ylabel="GA wall-clock time (s), mean ± SD of 5",
           title="GA wall-clock time, 600 generations")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    save(fig, "figure6_scalability.png")


if __name__ == "__main__":
    fig1_forecast()
    net, runs, table = seed0_runs()
    fig2_routes(net, table)
    fig3_convergence(runs)
    fig4_ablation()
    fig5_tradeoff()
    fig6_scalability()
    print("figures written to", FIGURES_DIR)
