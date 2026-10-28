"""Forecast accuracy at the routing decision point (Section 5.2).

Builds the disruption scenario (Section 4.2), forecasts the travel-time matrix for
17:00 on day 40 from the six preceding hours, and compares it with ground truth
separately for the 156 arcs incident to a disrupted node and the other 600 arcs.
The forecast is clipped as used for routing (Eq. 9).
"""
import numpy as np

from gtst_common import DECISION_DAY, DECISION_HOUR, DISRUPTED_NODES, Network, disrupted_scenario, forecast_matrix, load_model, save_json


def main():
    net = Network()
    model = load_model()
    t0 = DECISION_DAY * 24 + DECISION_HOUR
    ratio, tt = disrupted_scenario(net)
    pred = forecast_matrix(net, model, ratio, t0)[net.edge_i, net.edge_j]
    truth = tt[t0]
    err = np.abs(pred - truth)
    hit = np.isin(net.edge_i, DISRUPTED_NODES) | np.isin(net.edge_j, DISRUPTED_NODES)
    out = {}
    for name, mask in (("disrupted_arcs", hit), ("other_arcs", ~hit)):
        out[name] = {"n": int(mask.sum()),
                     "MAE_min": round(float(err[mask].mean()), 3),
                     "MAPE_pct": round(float((err[mask] / np.clip(truth[mask], 1e-3, None)).mean() * 100), 3),
                     "mean_forecast_over_truth": round(float((pred[mask] / truth[mask]).mean()), 3)}
        print(name, out[name])
    out["min_forecast_ratio_to_free_flow"] = round(float((pred / net.free_flow_vec).min()), 3)
    out["share_arcs_forecast_above_free_flow"] = float((pred > net.free_flow_vec).mean())
    save_json(out, "disruption_forecast_accuracy.json")


if __name__ == "__main__":
    main()
