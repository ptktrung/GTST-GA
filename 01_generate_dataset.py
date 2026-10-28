"""Generate the 45-day synthetic HCMC traffic dataset (Section 4.1, Eqs. 20-21).

Output: data/hcmc_traffic_dynamic.csv, one row per arc and hour
(45 days x 24 hours x 756 arcs = 816,480 rows). Deterministic for the fixed seed.

travel_time_min is the ground-truth travel time used for training and evaluation.
congestion_penalty_factor and incident_flag are descriptive columns and are not
used by the experiments.
"""
import os

import numpy as np
import pandas as pd

from gtst_common import DATA_DIR, SUSCEPTIBILITY, TRAFFIC_CSV

SEED = 42
FREE_FLOW_SPEED_KMH = 32.0
N_DAYS, HOURS_PER_DAY = 45, 24

# Rush-hour peaks: (centre hour, std in hours, weekday amplitude)
AM_PEAK = (7.75, 1.1, 1.6)
PM_PEAK = (17.5, 1.3, 2.0)
WEEKEND_FACTOR = 0.45            # weekend peak amplitude relative to weekdays

NOISE_SIGMA = 0.09               # multiplicative log-normal noise per arc-hour
INCIDENT_PROB = 0.003            # incident start probability per arc-hour
INCIDENT_MULT = (1.4, 2.8)
INCIDENT_HOURS = (1, 3)
MIN_RATIO = 0.85                 # lower bound on travel time / free-flow time


def diurnal_multiplier(hour, weekend):
    """m(t) of Eq. (21)."""
    peak = sum(a * np.exp(-0.5 * ((hour - c) / s) ** 2) for c, s, a in (AM_PEAK, PM_PEAK))
    return 1.0 + (WEEKEND_FACTOR * peak if weekend else peak)


def generate():
    rng = np.random.default_rng(SEED)
    nodes = pd.read_csv(os.path.join(DATA_DIR, "hcmc_nodes.csv"))
    edges = pd.read_csv(os.path.join(DATA_DIR, "hcmc_edges_static.csv"))
    n = len(nodes)
    dist = np.zeros((n, n))
    dist[edges.source_node.values, edges.target_node.values] = edges.physical_distance_km.values
    kappa = nodes["node_type"].map(SUSCEPTIBILITY).fillna(0.7).values
    free_flow = dist / FREE_FLOW_SPEED_KMH * 60.0

    ei, ej = np.where(dist > 0)                 # same order as hcmc_edges_static.csv
    n_arcs = len(ei)
    kappa_bar = 0.5 * (kappa[ei] + kappa[ej])
    ff = free_flow[ei, ej]
    incidents = {}                              # arc index -> (last hour, multiplier)
    frames = []
    for day in range(N_DAYS):
        weekend = (day % 7) in (5, 6)
        for hour in range(HOURS_PER_DAY):
            t = day * HOURS_PER_DAY + hour
            arc_mult = 1.0 + (diurnal_multiplier(hour, weekend) - 1.0) * kappa_bar
            noise = rng.lognormal(mean=0.0, sigma=NOISE_SIGMA, size=n_arcs)
            inc = np.ones(n_arcs)
            for k in range(n_arcs):
                if k in incidents and incidents[k][0] >= t:
                    inc[k] = incidents[k][1]
                elif rng.random() < INCIDENT_PROB:
                    mult = rng.uniform(*INCIDENT_MULT)
                    incidents[k] = (t + rng.integers(*INCIDENT_HOURS, endpoint=True), mult)
                    inc[k] = mult
            tt = np.maximum(ff * arc_mult * noise * inc, MIN_RATIO * ff)       # Eq. (20)
            frames.append(pd.DataFrame({
                "day": day, "hour": hour, "time_step": f"D{day:02d}T{hour:02d}:00",
                "source_node": ei, "target_node": ej,
                "travel_time_min": np.round(tt, 2),
                "congestion_penalty_factor": np.round(np.clip((tt - ff) / np.maximum(ff, 1e-6), 0, None), 3),
                "incident_flag": (inc > 1.0).astype(int),
                "free_flow_time_min": np.round(ff, 2),
            }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv(TRAFFIC_CSV, index=False)
    print(f"{len(df):,} rows written to {TRAFFIC_CSV}")
    print(f"mean travel time {df.travel_time_min.mean():.2f} min, incident share {100 * df.incident_flag.mean():.2f}% of arc-hours")
