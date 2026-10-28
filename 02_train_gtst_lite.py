"""Train and test GTST-Lite (Sections 3.2 and 5.1), or one of its ablation variants (Section 5.8).

    python 02_train_gtst_lite.py --evaluate           # test accuracy of the released weights (Table 2)
    python 02_train_gtst_lite.py --ablation full      # component ablation (Table 8): full | attn | conv | graph
    python 02_train_gtst_lite.py --out FILE           # train the full model and save its weights to FILE

models/gtst_lite.pt holds the released weights used for every routing result in Section 5.
They were trained with this script in an earlier environment. Retraining with a different
PyTorch build or BLAS library gives a slightly different model (on our current CPU setup:
test MAE 2.304 instead of 2.313), and routing results depend on the instance (Section 5.11).
For the ablation, the full model and the three variants are all trained in the same run
environment so that Table 8 compares like with like.

Training data: days 0-30; validation: days 31-37; test: days 38-44. Each sample is one arc at
one hour t, with inputs taken from hours t-6 .. t-1 only.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gtst_common import (DATA_DIR, LAG, MODEL_PATH, RESULTS_DIR, SUSCEPTIBILITY, TEST_DAYS, TRAFFIC_CSV, TRAIN_DAYS, VAL_DAYS,
                         GTSTLite, calendar_features, get_device, save_json)

EPOCHS, BATCH, LR, WEIGHT_DECAY = 18, 8192, 2e-3, 1e-5

# Loss history recorded when the released weights were trained (used for Figure 1, left).
TRAINING_LOG = os.path.join(os.path.dirname(MODEL_PATH), "gtst_lite_training_log.json")


def build_samples():
    """Feature arrays for every (hour, arc) with t >= LAG. Computed in float64, stored as float32."""
    nodes = pd.read_csv(os.path.join(DATA_DIR, "hcmc_nodes.csv"))
    edges = pd.read_csv(os.path.join(DATA_DIR, "hcmc_edges_static.csv"))
    dyn = pd.read_csv(TRAFFIC_CSV)
    n_nodes, n_edges = len(nodes), len(edges)
    ei, ej = edges.source_node.values.astype(int), edges.target_node.values.astype(int)
    n_t = dyn["day"].nunique() * dyn["hour"].nunique()
    assert (dyn.source_node.values[:n_edges] == ei).all() and (dyn.target_node.values[:n_edges] == ej).all()

    tt = dyn["travel_time_min"].values.reshape(n_t, n_edges)
    ff = dyn["free_flow_time_min"].values.reshape(n_t, n_edges)[0]
    ratio = tt / ff[None, :]

    kappa = nodes["node_type"].map(SUSCEPTIBILITY).fillna(0.7).values.astype(np.float32)
    dist = np.zeros((n_nodes, n_nodes))
    dist[ei, ej] = edges.physical_distance_km.values
    with np.errstate(divide="ignore"):
        inv = np.where(dist > 0, 1.0 / dist, 0.0)
    row = inv.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    P = (inv / row).astype(np.float32)
    outdeg = np.bincount(ei, minlength=n_nodes).astype(np.float64)
    out_mean = np.zeros((n_nodes, n_edges), dtype=np.float32)
    out_mean[ei, np.arange(n_edges)] = 1.0 / outdeg[ei]
    s = ratio.astype(np.float32) @ out_mean.T          # node congestion state per hour
    s_prop = s @ P.T
    log_d = np.log1p(dist[ei, ej]).astype(np.float32)
    log_d = (log_d - log_d.mean()) / (log_d.std() + 1e-6)
    static = np.stack([log_d, kappa[ei], kappa[ej]], axis=1)

    hours = range(LAG, n_t)
    X = {
        "lag": np.concatenate([ratio[t - LAG:t, :].T for t in hours]),
        "cal": np.concatenate([np.tile(calendar_features(t), (n_edges, 1)) for t in hours]),
        "static": np.concatenate([static for _ in hours]),
        "node_i": np.concatenate([np.stack([s[t - 1], s_prop[t - 1]], axis=1)[ei] for t in hours]),
        "node_j": np.concatenate([np.stack([s[t - 1], s_prop[t - 1]], axis=1)[ej] for t in hours]),
        "y": np.concatenate([ratio[t, :] for t in hours]),
        "ff": np.tile(ff, n_t - LAG),
    }
    X = {k: v.astype(np.float32) for k, v in X.items()}
    day = np.concatenate([np.full(n_edges, t // 24) for t in hours])
    return X, day


def evaluate(model, d):
    """MAE, RMSE, MAPE of travel time in minutes (Eq. 10), before the routing-time clipping."""
    model.eval()
    with torch.no_grad():
        pred = torch.exp(model(d["lag"], d["cal"], d["static"], d["node_i"], d["node_j"])) * d["ff"]
        actual = d["y"] * d["ff"]
        err = (pred - actual).cpu().numpy()
        act = actual.cpu().numpy()
    return {"MAE_min": round(float(np.mean(np.abs(err))), 3),
            "RMSE_min": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "MAPE_pct": round(float(np.mean(np.abs(err) / np.clip(act, 1e-3, None)) * 100), 3)}, pred.cpu().numpy(), act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true", help="evaluate the released weights only")
    ap.add_argument("--ablation", choices=["full", "attn", "conv", "graph"], default=None,
                    help="train one variant for Table 8 (attn/conv/graph = that branch disabled)")
    ap.add_argument("--out", default=None, help="train the full model and save its weights here")
    args = ap.parse_args()
    if not (args.evaluate or args.ablation or args.out):
        ap.error("choose --evaluate, --ablation or --out (the released weights are never overwritten)")

    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()
    X, day = build_samples()
    splits = {name: np.isin(day, list(days)) for name, days in
              (("train", TRAIN_DAYS), ("val", VAL_DAYS), ("test", TEST_DAYS))}
    data = {name: {k: torch.tensor(v[m]).to(device) for k, v in X.items()} for name, m in splits.items()}

    if args.evaluate:
        model = GTSTLite().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        test_metrics, pred, actual = evaluate(model, data["test"])
        print("released weights, test:", test_metrics)
        np.save(os.path.join(RESULTS_DIR, "pred_vs_actual_test.npy"), np.stack([pred, actual], axis=1))
        save_json({"n_train": int(splits["train"].sum()), "n_val": int(splits["val"].sum()),
                   "n_test": int(splits["test"].sum()), "n_parameters": sum(p.numel() for p in model.parameters()),
                   "test": test_metrics, "training_log": json.load(open(TRAINING_LOG))}, "forecast_validation.json")
        return

    disabled = [args.ablation] if args.ablation in ("attn", "conv", "graph") else []
    model = GTSTLite(disable=disabled).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    target = lambda d: torch.log(d["y"].clamp(min=1e-3))                # loss on the log-ratio, Eq. (8)
    n_train = data["train"]["y"].shape[0]
    history, t0 = [], time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total = 0.0
        for start in range(0, n_train, BATCH):
            idx = perm[start:start + BATCH]
            b = {k: v[idx] for k, v in data["train"].items()}
            opt.zero_grad()
            loss = mse(model(b["lag"], b["cal"], b["static"], b["node_i"], b["node_j"]), target(b))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            v = data["val"]
            val_loss = mse(model(v["lag"], v["cal"], v["static"], v["node_i"], v["node_j"]), target(v)).item()
        history.append((epoch, total / n_train, val_loss))
        print(f"epoch {epoch:2d}  train {total / n_train:.4f}  val {val_loss:.4f}")
    train_time = time.time() - t0

    test_metrics, pred, actual = evaluate(model, data["test"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"test: {test_metrics}  ({n_params} parameters, {train_time:.1f} s on {device})")

    if args.ablation:
        save_json({"variant": args.ablation, **test_metrics, "n_parameters": n_params,
                   "train_time_sec": round(train_time, 1), "history": history},
                  f"forecast_ablation_{args.ablation}.json")
    if args.out:
        torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
