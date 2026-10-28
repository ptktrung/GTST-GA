"""Wall-clock time of the GA by instance size (Section 5.7, Table 7, Figure 6).

Random instances of 28, 50, 75 and 100 nodes are drawn in the HCMC bounding box (padded by
15%), with Haversine distances, free-flow times at 32 km/h, a uniform congestion multiplier in
[1.3, 2.4] and log-normal noise (sigma 0.09). The objective and GA settings are those of the
main experiments. Five instances per size; only the GA run is timed. Run on an idle machine.

    python 11_scalability.py            # all sizes
    python 11_scalability.py 50 100     # selected sizes, then `python 11_scalability.py summary`
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from gtst_common import ALPHA, BETA, DATA_DIR, GENERATIONS, RESULTS_DIR, run_ga, save_json, seeded_population

SIZES = (28, 50, 75, 100)
REPEATS = 5
OUT_DIR = os.path.join(RESULTS_DIR, "scalability")


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin(np.radians(lat2 - lat1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def random_instance(n, seed):
    nodes = pd.read_csv(os.path.join(DATA_DIR, "hcmc_nodes.csv"))
    pad_lat = (nodes.latitude.max() - nodes.latitude.min()) * 0.15
    pad_lon = (nodes.longitude.max() - nodes.longitude.min()) * 0.15
    rs = np.random.RandomState(seed)
    lats = rs.uniform(nodes.latitude.min() - pad_lat, nodes.latitude.max() + pad_lat, size=n)
    lons = rs.uniform(nodes.longitude.min() - pad_lon, nodes.longitude.max() + pad_lon, size=n)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dist[i, :] = haversine_km(lats[i], lons[i], lats, lons)
    free = dist / 32.0 * 60.0
    times = free * (1.0 + rs.uniform(0.3, 1.4, size=(n, n))) * rs.lognormal(0.0, 0.09, size=(n, n))
    for m in (times, free, dist):
        np.fill_diagonal(m, 0.0)
    return dist.astype(np.float32), free.astype(np.float32), times.astype(np.float32)


def costs(route, dist, free, times):
    full = np.concatenate(([0], route.astype(int), [0]))
    src, tgt = full[:-1], full[1:]
    t = times[src, tgt]
    e = ALPHA * dist[src, tgt] + BETA * np.clip(t - free[src, tgt], 0, None)
    return t.sum(), e.sum()


def greedy(n, dist, free, times):
    unvisited, current, route = set(range(1, n)), 0, []
    while unvisited:
        nxt = min(unvisited, key=lambda k: (times[current, k] + ALPHA * dist[current, k]
                                            + BETA * max(0.0, times[current, k] - free[current, k]), k))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return np.array(route)


def run_size(n):
    secs, objective = [], []
    for rep in range(REPEATS):
        dist, free, times = random_instance(n, seed=1000 * n + rep)
        init = seeded_population(greedy(n, dist, free, times), rep, n)
        fitness = lambda r: 1.0 / (sum(costs(r, dist, free, times)) + 1e-6)
        start = time.time()
        best = run_ga(fitness, n, rep, initial_population=init, generations=GENERATIONS)
        secs.append(time.time() - start)
        objective.append(float(sum(costs(best, dist, free, times))))
        print(f"n = {n:3d}  repeat {rep + 1}: {secs[-1]:6.2f} s")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"n{n}.json"), "w") as f:
        json.dump({"seconds": secs, "objective": objective}, f, indent=2)


def summary():
    rows = {}
    for n in SIZES:
        r = json.load(open(os.path.join(OUT_DIR, f"n{n}.json")))
        rows[str(n)] = {"mean_sec": float(np.mean(r["seconds"])), "sd_sec": float(np.std(r["seconds"])),
                        "search_space": f"{n - 1}!"}
        print(f"n = {n:3d}: {rows[str(n)]['mean_sec']:.2f} ± {rows[str(n)]['sd_sec']:.2f} s")
    logn = np.log([int(n) for n in rows]); logt = np.log([v["mean_sec"] for v in rows.values()])
    save_json({"sizes": rows, "loglog_slope": float(np.polyfit(logn, logt, 1)[0])}, "scalability.json")


if __name__ == "__main__":
    args = sys.argv[1:] or [str(n) for n in SIZES] + ["summary"]
    for a in args:
        if a != "summary":
            run_size(int(a))
    if "summary" in args:
        summary()
