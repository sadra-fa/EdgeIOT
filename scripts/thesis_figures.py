#!/usr/bin/env python3
"""
thesis_figures.py — Print versions of the pipeline figures for the dissertation.

Recomputes, with the pipeline's own classes and configuration, the inputs of the
three upstream figures (standardised readings, PCA + Ward clusters, Isolation
Forest decision values) and draws them in the light style of the other figures,
with categorical cluster colours and a legend. Checks that the cluster sizes and
the number of flagged windows equal those of the stored pipeline run.

Writes dissertation/latex/figures/pipeline/{readings,clusters_pca,anomaly_scores}.pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgeiot.config import PipelineConfig                                           # noqa: E402
from edgeiot.data import DataLoader, DataPreprocessor, HierarchicalClusteringManager  # noqa: E402
from edgeiot.models import IsolationForestAnomalyDetector, SequenceWindowGenerator    # noqa: E402
from edgeiot.semantic.llm_reasoner import CLUSTER_NAMES                               # noqa: E402

OUT = PROJECT_ROOT / "dissertation" / "latex" / "figures" / "pipeline"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#ecebe7", "#d8d6d0"
FLAG = "#e34948"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    run = sorted((PROJECT_ROOT / "outputs" / "rerun").glob("*"))[-1]
    stored = json.loads((run / "pipeline_metrics.json").read_text())

    df = DataLoader(str(PROJECT_ROOT / "RealLife.csv")).load_and_clean()
    X = DataPreprocessor(cfg.preprocessing).preprocess(df)

    # 1. Standardised readings in file order (first 200 rows)
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 3.6), sharex=True)
    for ax, col, colour in zip(axes, ["temperature", "humidity", "noise_db"], SERIES):
        ax.plot(np.arange(200), X[col].values[:200], color=colour, linewidth=1.2)
        ax.set_ylabel(col.replace("_db", ""), fontsize=8, color=INK2)
        style(ax)
    axes[-1].set_xlabel("Row in file", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "readings.pdf")
    plt.close(fig)

    # 2. PCA projection coloured by Ward cluster
    cache = run / "thesis_figures_cache.npz"     # Ward labels and IF scores, to redraw quickly
    if cache.exists():
        saved = np.load(cache, allow_pickle=True)
        labels, scores, flags = saved["labels"], saved["scores"], saved["flags"]
    else:
        labels = HierarchicalClusteringManager(cfg.clustering).fit_predict(X)
        X_seq = SequenceWindowGenerator(cfg.preprocessing.sensor_cols, cfg.anomaly.window_size).transform(X)
        det = IsolationForestAnomalyDetector(cfg.anomaly)
        det.fit_predict(X_seq)
        scores, flags = det.scores, det.labels
        np.savez(cache, labels=labels, scores=scores, flags=flags)
    sizes = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    assert sizes == stored["cluster_sizes"], (sizes, stored["cluster_sizes"])
    Z = PCA(n_components=2).fit_transform(X.values)
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    rng = np.random.default_rng(42)
    order = rng.permutation(len(Z))          # no cluster is drawn systematically on top
    for k in range(3):
        idx = order[labels[order] == k]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=3, color=SERIES[k], alpha=0.5, linewidths=0,
                   label=f"{k} = {CLUSTER_NAMES[k]} ({sizes[str(k)]:,})")
    ax.set_xlabel("PC1", fontsize=8.5, color=INK2)
    ax.set_ylabel("PC2", fontsize=8.5, color=INK2)
    style(ax)
    ax.legend(fontsize=8, frameon=False, markerscale=4, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "clusters_pca.pdf", dpi=200)
    plt.close(fig)

    # 3. Isolation Forest decision values per window
    flagged = flags == "ABNORMAL"
    assert int(flagged.sum()) == stored["anomaly_flagged"]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    x = np.arange(len(scores))
    ax.scatter(x[~flagged], scores[~flagged], s=1.5, color=SERIES[0], alpha=0.35, linewidths=0,
               label=f"normal ({(~flagged).sum():,})")
    ax.scatter(x[flagged], scores[flagged], s=2.5, color=FLAG, alpha=0.8, linewidths=0,
               label=f"flagged ({flagged.sum():,}, 5%)")
    ax.axhline(0, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Window (file order)", fontsize=8.5, color=INK2)
    ax.set_ylabel("Decision value", fontsize=8.5, color=INK2)
    style(ax)
    ax.legend(fontsize=8, frameon=False, markerscale=5, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "anomaly_scores.pdf", dpi=200)
    plt.close(fig)
    print("figures written; cluster sizes and flagged windows match the stored run")


if __name__ == "__main__":
    main()
