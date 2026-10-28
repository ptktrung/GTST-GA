# GTST-GA: forecast-driven tour optimization under disruption

Code, data, trained weights and per-run results for the paper

> T. K. T. Pham, V. V. Le, and D. A. Dewi, "Dynamic Routing of Autonomous Transit in Stochastic Mixed-Mobility Networks: A Hybrid Graph-Theoretic Forecasting and Evolutionary Approach," submitted to *Journal of Intelligent Transportation Systems*.

The pipeline has two stages. GTST-Lite, a small spatiotemporal model with 2,345 parameters, forecasts the next-hour travel time of every arc of a 28-node Ho Chi Minh City network. A memetic genetic algorithm then sequences one vehicle through all nodes, minimizing travel time plus a congestion-based energy index computed on the forecast. Every tour is scored on ground-truth travel times.

## Contents

```
gtst_common.py                     shared: data, GTST-Lite model, forecast, disruption scenario, costs, GA
01_generate_dataset.py             synthetic 45-day traffic data                      Sec. 4.1
02_train_gtst_lite.py              evaluate released model / train / ablation         Sec. 5.1, 5.8 (Tables 2, 8)
03_forecast_baselines.py           persistence and historical-average baselines       Table 2
04_disruption_forecast_accuracy.py forecast error on disrupted vs other arcs          Sec. 5.2
05_compare_methods.py              N = 30 comparison of all routing methods           Tables 3-4, Sec. 5.3-5.4
06_statistics.py                   tests, effect sizes, intervals                     Table 11
07_tradeoff_sweep.py               weighted-sum time/energy sweep                     Table 5, Figure 5
08_beta_sensitivity.py             congestion-penalty sensitivity                     Table 6
09_speed_term.py                   speed-dependent energy term                        Table 9
10_rolling_horizon.py              two-cycle re-planning, warm vs cold start          Table 10
11_scalability.py                  GA wall-clock time, 28-100 nodes                   Table 7, Figure 6
12_distance_lower_bound.py         1-tree lower bound on tour distance                Sec. 5.11
13_figures.py                      Figures 1-6 and the seed-0 table                   Table 3, Figures 1-6
run_all.sh                         runs every step in order
data/                              nodes, arcs, hourly traffic (816,480 rows)
models/                            released GTST-Lite weights and their training log
results/                           outputs of every script, including per-run results
figures/                           figures as used in the paper
```

## Installation

Python 3.10 or later. The results in the paper were produced with Python 3.13.17, PyTorch 2.14.1 (CPU), pygad 3.7.0, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, networkx and matplotlib.

```
pip install -r requirements.txt
```

## Reproducing the results

```
export GTST_FORCE_CPU=1      # CPU execution, as used for every reported number
bash run_all.sh
```

Each script can also be run on its own. All scripts read from `data/` and `models/` and write to `results/` and `figures/`. On one CPU core, the full run takes a few hours, most of it in `05_compare_methods.py` and `07_tradeoff_sweep.py`. Methods can be run in parallel, for example `python 05_compare_methods.py GTST-GA Oracle-GA` in one shell and `python 05_compare_methods.py Static-GA` in another, followed by `python 05_compare_methods.py summary`.

`11_scalability.py` measures wall-clock time, so it should be run on an otherwise idle machine. Absolute times depend on hardware.

## Notes

- **Released weights.** `models/gtst_lite.pt` is the forecaster behind every routing result. `02_train_gtst_lite.py` never overwrites it: use `--out FILE` to train a new model. Training is deterministic within one software environment, but not across PyTorch builds or BLAS libraries. On our current CPU setup, retraining gives a test MAE of 2.304 min instead of 2.313 min. Routing results depend on the forecaster instance (paper, Section 5.11), so use the released weights to reproduce the reported numbers.
- **Component ablation (Table 8).** The full model and the three variants are all retrained in the same environment (`--ablation full|attn|conv|graph`), so that the comparison is like for like.
- **Tour cost.** Every leg of a tour is costed with the forecast for the decision hour, 17:00 on day 40 (paper, Eq. 11). Time-dependent leg costs (paper, Eq. 12) are not implemented.
- **Disruption.** Arcs with one end at node 0, 2 or 22 are scaled by 4 from 15:00 to 17:00, and the six arcs joining two of these nodes by 16 (`gtst_common.disrupted_scenario`).
- **Energy index.** `C_energy = 1.2 d + 2.5 max(0, tau - tau_ff)` is a congestion-weighted index in kWh-equivalent units, not physical battery energy (paper, Section 3.3.2).
- **Genetic algorithm.** pygad with a population of 150, 600 generations, tournament selection, single-point crossover with duplicate-gene repair, inversion mutation and 5 elite tours (paper, Table 1). Seed `k` of each method is run `k`.
- **Randomness.** All seeds are fixed. Results are reproducible on the same software stack. Different library versions or hardware can change GA trajectories.

## Data

- `data/hcmc_nodes.csv`: 28 nodes (depot, 21 districts and counties, 6 transit hubs) with coordinates and node type.
- `data/hcmc_edges_static.csv`: 756 directed arcs with Haversine distances in km.
- `data/hcmc_traffic_dynamic.csv`: hourly ground-truth travel times (`travel_time_min`) and free-flow times for 45 days, generated by `01_generate_dataset.py`. The columns `congestion_penalty_factor` and `incident_flag` are descriptive and are not used by the experiments.

The traffic data are synthetic. The generator and its parameters are described in Section 4.1 of the paper.
