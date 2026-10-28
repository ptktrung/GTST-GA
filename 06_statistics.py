"""Statistical comparison with GTST-GA on travel time (Table 11, Section 4.4).

Reads results/comparison/*.json. For each method: mean, sample SD, 95% t-interval,
median and IQR; against GTST-GA: two-sided Wilcoxon signed-rank test (runs paired by
seed index), two-sided Mann-Whitney U test (unpaired), Holm-adjusted p-values over the
five comparisons, Cohen's d with pooled sample SD (Eq. 23; negative = GTST-GA faster),
and a 95% percentile-bootstrap interval for the difference in means (10,000 resamples).
"""
import json
import os

import numpy as np
from scipy import stats

from gtst_common import RESULTS_DIR, save_json

COMPARED = ["Static-GA", "Oracle-GA", "Ablation-NoMemetic", "Ablation-SwapMutation", "2opt-Oropt"]


def holm(pvals):
    order = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = running
    return adj


def main():
    load = lambda m: json.load(open(os.path.join(RESULTS_DIR, "comparison", f"{m}.json")))
    g = np.array(load("GTST-GA")["time"])
    n = len(g)
    rng = np.random.default_rng(12345)
    out = {}
    for m in ["GTST-GA"] + COMPARED:
        x = np.array(load(m)["time"])
        half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        row = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "ci95": [float(x.mean() - half), float(x.mean() + half)],
               "median": float(np.median(x)), "iqr": [float(np.percentile(x, 25)), float(np.percentile(x, 75))]}
        if m != "GTST-GA":
            sp = np.sqrt(((n - 1) * g.var(ddof=1) + (len(x) - 1) * x.var(ddof=1)) / (n + len(x) - 2))
            boot = [rng.choice(g, n).mean() - rng.choice(x, len(x)).mean() for _ in range(10000)]
            row.update({"diff_gtst_minus_method": float(g.mean() - x.mean()),
                        "diff_boot_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "cohens_d": float((g.mean() - x.mean()) / sp),
                        "wilcoxon_p": float(stats.wilcoxon(g, x).pvalue),
                        "mannwhitney_p": float(stats.mannwhitneyu(g, x, alternative="two-sided").pvalue)})
        out[m] = row
    for test in ("wilcoxon_p", "mannwhitney_p"):
        for m, p in holm({m: out[m][test] for m in COMPARED}).items():
            out[m][test + "_holm"] = p
    for m in COMPARED:
        r = out[m]
        print(f"{m:24s} diff {r['diff_gtst_minus_method']:+7.1f} [{r['diff_boot_ci95'][0]:+.1f}, {r['diff_boot_ci95'][1]:+.1f}]"
              f"  d {r['cohens_d']:+.2f}  Wilcoxon {r['wilcoxon_p']:.4f} (Holm {r['wilcoxon_p_holm']:.4f})"
              f"  Mann-Whitney {r['mannwhitney_p']:.4f} (Holm {r['mannwhitney_p_holm']:.4f})")
    save_json(out, "comparison_statistics.json")


if __name__ == "__main__":
    main()
