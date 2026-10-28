"""Shared definitions for the GTST-GA experiments.

Contents
    - constants (data splits, energy coefficients, disruption scenario, GA settings)
    - Network: static graph and hourly travel-time data
    - GTSTLite: the forecasting model (Section 3.2 of the paper)
    - forecast_matrix(): one-hour-ahead travel-time forecast for every arc
    - disrupted_scenario(): the 400% disruption of Section 4.2
    - route_costs(), nearest_neighbor(), run_ga(): routing objective and optimizer (Sections 3.3-3.4)

Equation and section numbers in comments refer to the paper.
"""
import json
import os

import numpy as np
import pandas as pd
import pygad
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
MODEL_PATH = os.path.join(MODEL_DIR, "gtst_lite.pt")
TRAFFIC_CSV = os.path.join(DATA_DIR, "hcmc_traffic_dynamic.csv")

# Forecasting
LAG = 6                                   # hours in the input window, Eq. (3)
TRAIN_DAYS = range(0, 31)
VAL_DAYS = range(31, 38)
TEST_DAYS = range(38, 45)
RATIO_CLIP = (0.5, 8.0)                   # Eq. (9)

# Energy index, Eq. (13)
ALPHA = 1.2                               # kWh per km, from Vepsalainen et al. (2019)
BETA = 2.5                                # kWh-equivalent per minute of delay (assumed)

# Node-type congestion susceptibility (Section 4.1)
SUSCEPTIBILITY = {"Depot": 1.00, "Transit Hub": 1.15, "District": 0.90, "County": 0.45}

# Disruption scenario (Section 4.2)
DISRUPTED_NODES = (0, 2, 22)
DECISION_DAY = 40
DECISION_HOUR = 17
LEAD_HOURS = 2
DISRUPTION_FACTOR = 4.0

# Genetic algorithm (Table 1)
POP_SIZE = 150
GENERATIONS = 600
PARENTS_MATING = 20
ELITISM = 5


def get_device():
    """CPU unless a GPU is available; set GTST_FORCE_CPU=1 to force CPU (used for all reported results)."""
    if os.environ.get("GTST_FORCE_CPU", "0") == "1" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def save_json(obj, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Network and data
# ---------------------------------------------------------------------------
class Network:
    """Static graph plus the hourly travel-time data (float32 throughout).

    Arrays indexed by arc follow the row order of hcmc_edges_static.csv,
    which is also the order of the arcs within each hour of the traffic file.
    """

    def __init__(self, traffic_csv=TRAFFIC_CSV):
        self.nodes = pd.read_csv(os.path.join(DATA_DIR, "hcmc_nodes.csv"))
        edges = pd.read_csv(os.path.join(DATA_DIR, "hcmc_edges_static.csv"))
        self.n_nodes = len(self.nodes)
        self.edge_i = edges.source_node.astype(int).values
        self.edge_j = edges.target_node.astype(int).values
        self.n_edges = len(edges)

        dyn = pd.read_csv(traffic_csv)
        self.n_t = dyn["day"].nunique() * dyn["hour"].nunique()
        self.travel_time = dyn["travel_time_min"].values.reshape(self.n_t, self.n_edges).astype(np.float32)
        self.free_flow_vec = dyn["free_flow_time_min"].values.reshape(self.n_t, self.n_edges)[0].astype(np.float32)
        self.ratio = self.travel_time / self.free_flow_vec[None, :]          # Eq. (2)

        self.dist = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        self.dist[self.edge_i, self.edge_j] = edges.physical_distance_km.values
        self.free_flow = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        self.free_flow[self.edge_i, self.edge_j] = self.free_flow_vec

        kappa = self.nodes["node_type"].map(SUSCEPTIBILITY).fillna(0.7).values.astype(np.float32)
        self.kappa_i = kappa[self.edge_i]
        self.kappa_j = kappa[self.edge_j]

        # Row-normalized inverse-distance propagation matrix, Eq. (5)
        with np.errstate(divide="ignore"):
            inv = np.where(self.dist > 0, 1.0 / self.dist, 0.0)
        row = inv.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        self.P = (inv / row).astype(np.float32)

        # Averaging operator from arcs to their source node (mean over outgoing arcs)
        outdeg = np.bincount(self.edge_i, minlength=self.n_nodes).astype(np.float64)
        self.out_mean = np.zeros((self.n_nodes, self.n_edges), dtype=np.float32)
        self.out_mean[self.edge_i, np.arange(self.n_edges)] = 1.0 / outdeg[self.edge_i]

        log_d = np.log1p(self.dist[self.edge_i, self.edge_j])
        self.log_dist_std = ((log_d - log_d.mean()) / (log_d.std() + 1e-6)).astype(np.float32)

    def to_matrix(self, arc_values):
        m = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        m[self.edge_i, self.edge_j] = arc_values
        return m


def calendar_features(t):
    """Eq. (4): cyclical hour-of-day and day-of-week encoding plus a weekend flag (day 0 is a Monday)."""
    day, hour = t // 24, t % 24
    dow = day % 7
    return [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            1.0 if dow in (5, 6) else 0.0]


# ---------------------------------------------------------------------------
# Forecasting model, Section 3.2
# ---------------------------------------------------------------------------
class GTSTLite(nn.Module):
    """Dilated-convolution, attention and graph-propagation branches feeding an MLP head.

    `disable` may contain "conv", "attn" and/or "graph"; a disabled branch
    contributes zeros (used for the component ablation, Section 5.8).
    Output: log of the predicted travel-time ratio, Eq. (7).
    """

    def __init__(self, lag_len=LAG, emb=8, disable=()):
        super().__init__()
        self.emb = emb
        self.disable = set(disable)
        self.conv1 = nn.Conv1d(1, emb, kernel_size=2, dilation=1)
        self.conv2 = nn.Conv1d(emb, emb, kernel_size=2, dilation=2)
        self.attn_proj = nn.Linear(1, emb)
        self.attn = nn.MultiheadAttention(embed_dim=emb, num_heads=1, batch_first=True)
        pos = torch.arange(lag_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, emb, 2).float() * (-np.log(10000.0) / emb))
        pe = torch.zeros(lag_len, emb)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe)
        self.node_mlp = nn.Sequential(nn.Linear(2, emb), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(4 * emb + 5 + 3, 32), nn.ReLU(),
                                  nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, lag, cal, static, node_i, node_j):
        b = lag.shape[0]
        zeros = lambda: torch.zeros(b, self.emb, device=lag.device)
        if "conv" in self.disable:
            h_conv = zeros()
        else:
            c = torch.relu(self.conv1(lag.unsqueeze(1)))
            h_conv = torch.relu(self.conv2(c)).mean(dim=-1)
        if "attn" in self.disable:
            h_att = zeros()
        else:
            h = self.attn_proj(lag.unsqueeze(-1)) + self.pos_enc.unsqueeze(0)
            h_att = self.attn(h, h, h)[0].mean(dim=1)                       # Eq. (6)
        if "graph" in self.disable:
            g_i, g_j = zeros(), zeros()
        else:
            g_i, g_j = self.node_mlp(node_i), self.node_mlp(node_j)
        feat = torch.cat([h_conv, h_att, g_i, g_j, cal, static], dim=-1)
        return self.head(feat).squeeze(-1)


def load_model(device=None):
    device = device or get_device()
    model = GTSTLite().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def forecast_matrix(net, model, ratio_hist, t, device=None):
    """Forecast travel times for hour t from the observed ratios of hours t-6..t-1.

    Returns the (n_nodes x n_nodes) matrix of Eq. (9), with the ratio clipped to RATIO_CLIP.
    """
    device = device or next(model.parameters()).device
    lag = ratio_hist[t - LAG:t, :].T
    s = ratio_hist[t - 1, :] @ net.out_mean.T                                # Eq. (5), s_i
    s_prop = s @ net.P.T                                                     # Eq. (5), P s
    cal = np.tile(calendar_features(t), (net.n_edges, 1)).astype(np.float32)
    static = np.stack([net.log_dist_std, net.kappa_i, net.kappa_j], axis=1).astype(np.float32)
    node_i = np.stack([s[net.edge_i], s_prop[net.edge_i]], axis=1).astype(np.float32)
    node_j = np.stack([s[net.edge_j], s_prop[net.edge_j]], axis=1).astype(np.float32)
    as_t = lambda a: torch.tensor(a, device=device)
    with torch.no_grad():
        log_ratio = model(as_t(lag.astype(np.float32)), as_t(cal), as_t(static), as_t(node_i), as_t(node_j))
        ratio = torch.exp(log_ratio).cpu().numpy()
    ratio = np.clip(ratio, *RATIO_CLIP)
    return net.to_matrix(ratio * net.free_flow_vec)


def disrupted_scenario(net, nodes=DISRUPTED_NODES, last_hour=None, lead=LEAD_HOURS, factor=DISRUPTION_FACTOR):
    """Copy of the ratio and travel-time histories with the disruption of Section 4.2 applied
    for hours last_hour-lead .. last_hour.

    The factor is applied once per disrupted end node: arcs with one disrupted end are
    scaled by `factor` (x4), and the six arcs joining two disrupted nodes by factor**2 (x16).
    """
    if last_hour is None:
        last_hour = DECISION_DAY * 24 + DECISION_HOUR
    ratio, tt = net.ratio.copy(), net.travel_time.copy()
    hours = list(range(last_hour - lead, last_hour + 1))
    n_ends = np.isin(net.edge_i, nodes).astype(int) + np.isin(net.edge_j, nodes).astype(int)
    for e in np.flatnonzero(n_ends):
        for _ in range(n_ends[e]):
            tt[hours, e] *= factor
        ratio[hours, e] = tt[hours, e] / net.free_flow_vec[e]
    return ratio, tt


# ---------------------------------------------------------------------------
# Routing objective and optimizer, Sections 3.3-3.4
# ---------------------------------------------------------------------------
def tour_arcs(route):
    full = np.concatenate(([0], np.asarray(route).astype(int), [0]))
    return full[:-1], full[1:]


def route_costs(net, route, time_mat, beta=BETA, as_float=False):
    """Distance, C_time (Eq. 11) and C_energy (Eq. 13) of a tour on a given travel-time matrix."""
    src, tgt = tour_arcs(route)
    d = net.dist[src, tgt]
    t = time_mat[src, tgt]
    excess = np.clip(t - net.free_flow[src, tgt], 0, None)
    e = ALPHA * d + beta * excess
    if as_float:
        return float(d.sum()), float(t.sum()), float(e.sum())
    return d.sum(), t.sum(), e.sum()


def nearest_neighbor(net, time_mat, beta=BETA):
    """Greedy tour using the per-arc contribution to F as the step cost, Eq. (19)."""
    unvisited, current, route = set(range(1, net.n_nodes)), 0, []
    while unvisited:
        best, best_cost = None, float("inf")
        for k in unvisited:
            excess = max(0.0, time_mat[current, k] - net.free_flow[current, k])
            cost = time_mat[current, k] + ALPHA * net.dist[current, k] + beta * excess
            if cost < best_cost:
                best, best_cost = k, cost
        route.append(best)
        unvisited.remove(best)
        current = best
    return np.array(route)


def seeded_population(seed_route, seed, n_nodes, size=POP_SIZE):
    """One given tour plus random permutations drawn from RandomState(seed)."""
    rs = np.random.RandomState(seed)
    return [list(seed_route)] + [rs.permutation(range(1, n_nodes)).tolist() for _ in range(size - 1)]


def run_ga(fitness, n_nodes, seed, initial_population=None, generations=GENERATIONS,
           mutation_type="inversion", on_generation=None):
    """pygad GA with the settings of Table 1. Returns the best tour (genes 1..n, depot excluded)."""
    ga = pygad.GA(
        num_generations=generations, num_parents_mating=PARENTS_MATING,
        fitness_func=lambda ga_instance, sol, idx: fitness(sol),
        initial_population=initial_population,
        sol_per_pop=POP_SIZE if initial_population is None else None,
        num_genes=n_nodes - 1, gene_type=int, gene_space=range(1, n_nodes), allow_duplicate_genes=False,
        parent_selection_type="tournament", crossover_type="single_point",
        mutation_type=mutation_type, mutation_percent_genes=20, keep_elitism=ELITISM,
        on_generation=on_generation, suppress_warnings=True, random_seed=seed,
    )
    ga.run()
    return ga.best_solution()[0]


def sum_fitness(net, time_mat, beta=BETA, as_float=False):
    """Fitness f(R) = 1 / (C_time + C_energy + 1e-6), Eq. (15)."""
    def f(route):
        _, t, e = route_costs(net, route, time_mat, beta=beta, as_float=as_float)
        return 1.0 / (t + e + 1e-6)
    return f
