#!/usr/bin/env bash
# Reproduce every result and figure of the paper on CPU (several hours on one core).
# The dataset and the trained weights are included; steps 01 and the --ablation runs of 02 can be skipped.
set -euo pipefail
export GTST_FORCE_CPU=1
python 01_generate_dataset.py
python 02_train_gtst_lite.py --evaluate
for v in full attn conv graph; do python 02_train_gtst_lite.py --ablation "$v"; done
python 03_forecast_baselines.py
python 04_disruption_forecast_accuracy.py
python 05_compare_methods.py
python 06_statistics.py
python 07_tradeoff_sweep.py
python 08_beta_sensitivity.py
python 09_speed_term.py
python 10_rolling_horizon.py
python 11_scalability.py          # timing: run on an otherwise idle machine
python 12_distance_lower_bound.py
python 13_figures.py
