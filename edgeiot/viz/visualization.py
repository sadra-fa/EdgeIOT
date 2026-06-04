import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from xgboost import plot_importance

logger = logging.getLogger(__name__)

# ── Dark theme palette ───────────────────────────────────────────────────────
DARK   = "#0f1117"
ACCENT = "#38bdf8"
WARN   = "#f87171"
OK     = "#4ade80"


class PipelineVisualizer:
    """Aggregates and modularises all plotting for the Smart Home Edge AI pipeline."""

    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = output_dir

    # ── helpers ───────────────────────────────────────────────────────────
    def _savefig(self, name: str) -> None:
        path = f"{self.output_dir}/{name}"
        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK)
        plt.close()
        logger.info(f"  saved → {path}")

    @staticmethod
    def _style_ax(ax) -> None:
        ax.set_facecolor("#1a1f2e")
        ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values():
            s.set_edgecolor("#334155")

    # ── 1  Sensor time-series ─────────────────────────────────────────────
    def plot_timeseries(
        self,
        df: pd.DataFrame,
        cols: List[str],
        n_samples: int = 200,
        fname: str = "01_data_exploration/timeseries.png",
    ) -> None:
        colors = [ACCENT, OK, WARN]
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor(DARK)
        self._style_ax(ax)
        for col, color in zip(cols[:3], colors):
            ax.plot(df[col][:n_samples].values, label=col, color=color, lw=0.9)
        ax.set_title(f"Smart Home Sensor Time-Series (first {n_samples} samples)", color="white")
        ax.set_xlabel("Time", color="#94a3b8")
        ax.set_ylabel("Normalised Value", color="#94a3b8")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        self._savefig(fname)

    # ── 2  PCA clustering scatter ─────────────────────────────────────────
    def plot_pca_clustering(
        self,
        X_pca: np.ndarray,
        labels: np.ndarray,
        fname: str = "02_clustering/clustering_pca.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(DARK)
        self._style_ax(ax)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="cool", s=15, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Cluster").ax.yaxis.label.set_color("#94a3b8")
        ax.set_title("Hierarchical Clustering (Ward) — PCA 2D", color="white")
        ax.set_xlabel("PC1", color="#94a3b8")
        ax.set_ylabel("PC2", color="#94a3b8")
        self._savefig(fname)

    # ── 3  Confusion matrix ──────────────────────────────────────────────
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "XGBoost Confusion Matrix",
        fname: str = "03_classification/confusion_matrix.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(DARK)
        self._style_ax(ax)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
        ax.set_title(title, color="white")
        self._savefig(fname)

    # ── 4  Feature importance ────────────────────────────────────────────
    def plot_feature_importance(
        self,
        xgb_model,
        max_features: int = 15,
        fname: str = "03_classification/feature_importance.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(DARK)
        self._style_ax(ax)
        plot_importance(xgb_model, ax=ax, max_num_features=max_features, color=ACCENT)
        ax.set_title("XGBoost Feature Importance", color="white")
        ax.set_xlabel("F-score", color="#94a3b8")
        ax.set_ylabel("Feature", color="#94a3b8")
        self._savefig(fname)

    # ── 5  Multi-class ROC ───────────────────────────────────────────────
    def plot_multiclass_roc(
        self,
        model,
        X_test: pd.DataFrame,
        y_test_bin: np.ndarray,
        model_name: str = "XGBoost",
        fname: str = "03_classification/roc.png",
    ) -> None:
        y_score = model.predict_proba(X_test)
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(DARK)
        self._style_ax(ax)
        colors = [ACCENT, OK, WARN]
        for i in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], ls="--", color="#475569")
        ax.set_xlabel("False Positive Rate", color="#94a3b8")
        ax.set_ylabel("True Positive Rate", color="#94a3b8")
        ax.set_title(f"ROC Curve — {model_name}", color="white")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        self._savefig(fname)

    # ── 6  Model comparison bar chart ────────────────────────────────────
    def plot_model_comparison(
        self,
        names: List[str],
        accuracies: List[float],
        fname: str = "03_classification/model_comparison.png",
    ) -> None:
        colors = [WARN, ACCENT, OK]
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(DARK)
        self._style_ax(ax)
        bars = ax.bar(names, accuracies, color=colors[: len(names)], width=0.5)
        ax.bar_label(bars, fmt="%.3f", color="white", fontsize=10, padding=4)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy", color="#94a3b8")
        ax.set_title("Model Performance Comparison", color="white")
        self._savefig(fname)

    # ── 7  SHAP beeswarm ────────────────────────────────────────────────
    def plot_shap_beeswarm(
        self,
        shap_values,
        class_idx: int = 0,
        max_display: int = 12,
        fname: str = "03_classification/shap_beeswarm.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(DARK)
        import shap as _shap
        _shap.plots.beeswarm(shap_values[:, :, class_idx], max_display=max_display, show=False)
        plt.title("SHAP Beeswarm — Class 0", color="white")
        self._savefig(fname)

    # ── 8  Anomaly scatter ───────────────────────────────────────────────
    def plot_anomaly_scores(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        fname: str = "04_anomaly_detection/anomaly.png",
    ) -> None:
        n_anom = (labels == "ABNORMAL").sum()
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor(DARK)
        self._style_ax(ax)
        colors = [WARN if x == "ABNORMAL" else ACCENT for x in labels]
        ax.scatter(range(len(scores)), scores, c=colors, s=8, alpha=0.7)
        ax.axhline(0, color="#475569", lw=0.8, ls="--")
        ax.set_title(
            f"Anomaly Detection — {n_anom} flagged ({n_anom / len(labels) * 100:.1f}%)",
            color="white",
        )
        ax.set_xlabel("Sequence index", color="#94a3b8")
        ax.set_ylabel("Anomaly score", color="#94a3b8")
        self._savefig(fname)

    # ── 9  Token distribution ────────────────────────────────────────────
    def plot_token_distribution(
        self,
        token_counts: Dict[str, int],
        fname: str = "05_tokenization/token_distribution.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(DARK)
        self._style_ax(ax)
        ax.bar(token_counts.keys(), token_counts.values(), color=ACCENT)
        plt.xticks(rotation=40, ha="right", color="#94a3b8")
        ax.set_title("Semantic Token Distribution", color="white")
        ax.set_ylabel("Frequency", color="#94a3b8")
        plt.tight_layout()
        self._savefig(fname)

    # ── 10  LLM benchmark dashboard (2×2) ────────────────────────────────
    def plot_llm_benchmark(
        self,
        df_bench: pd.DataFrame,
        suptitle: str = "Llama3.2:3b — Jetson Orin Nano Benchmark",
        fname: str = "06_llm_benchmark/llm_benchmark.png",
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.patch.set_facecolor(DARK)

        # Latency timeline
        ax = axes[0, 0]; self._style_ax(ax)
        ax.plot(df_bench["latency_ms"], color=ACCENT, lw=0.9)
        ax.axhline(df_bench["latency_ms"].mean(), color=WARN, ls="--",
                    label=f"Mean: {df_bench['latency_ms'].mean():.0f} ms")
        ax.axhline(df_bench["latency_ms"].quantile(0.95), color=OK, ls=":",
                    label=f"P95: {df_bench['latency_ms'].quantile(0.95):.0f} ms")
        ax.set_title("Inference Latency Over Time", color="white")
        ax.set_xlabel("Sample", color="#94a3b8")
        ax.set_ylabel("ms", color="#94a3b8")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)

        # Latency histogram
        ax = axes[0, 1]; self._style_ax(ax)
        ax.hist(df_bench["latency_ms"], bins=30, color=ACCENT, edgecolor="none", alpha=0.8)
        ax.set_title("Latency Distribution", color="white")
        ax.set_xlabel("ms", color="#94a3b8")
        ax.set_ylabel("Count", color="#94a3b8")

        # Tokens/sec by behaviour
        ax = axes[1, 0]; self._style_ax(ax)
        for beh, col in zip(["active", "sleeping", "away"], [OK, ACCENT, WARN]):
            subset = df_bench[df_bench["behaviour"] == beh]["tokens_per_sec"]
            if len(subset):
                ax.hist(subset, bins=15, color=col, alpha=0.7, label=beh)
        ax.set_title("Tokens/sec by Behaviour", color="white")
        ax.set_xlabel("tok/s", color="#94a3b8")
        ax.set_ylabel("Count", color="#94a3b8")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)

        # Latency: normal vs anomaly
        ax = axes[1, 1]; self._style_ax(ax)
        normal_lat = df_bench[~df_bench["is_anomaly"]]["latency_ms"]
        anom_lat   = df_bench[df_bench["is_anomaly"]]["latency_ms"]
        ax.boxplot(
            [normal_lat, anom_lat],
            labels=["Normal", "Anomaly"],
            patch_artist=True,
            boxprops=dict(facecolor="#1e293b", color=ACCENT),
            medianprops=dict(color=WARN),
            whiskerprops=dict(color="#94a3b8"),
            capprops=dict(color="#94a3b8"),
        )
        ax.set_title("Latency: Normal vs Anomaly Samples", color="white")
        ax.set_ylabel("ms", color="#94a3b8")

        fig.suptitle(suptitle, color="white", fontsize=13)
        plt.tight_layout()
        self._savefig(fname)

    # ── 11  Token efficiency comparison ──────────────────────────────────
    def plot_token_efficiency(
        self,
        df_eff: pd.DataFrame,
        fname: str = "06_llm_benchmark/token_efficiency.png",
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(DARK)

        ax = axes[0]; self._style_ax(ax)
        x = range(len(df_eff))
        ax.plot(list(x), df_eff["raw_latency_ms"].to_numpy(), color=WARN, lw=1.2, label="Raw sensor values")
        ax.plot(list(x), df_eff["sem_latency_ms"].to_numpy(), color=OK,   lw=1.2, label="Semantic tokens")
        ax.set_title("Latency: Raw vs Semantic Input", color="white")
        ax.set_xlabel("Sample", color="#94a3b8")
        ax.set_ylabel("ms", color="#94a3b8")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")

        ax = axes[1]; self._style_ax(ax)
        means = [df_eff["raw_latency_ms"].mean(), df_eff["sem_latency_ms"].mean()]
        bars  = ax.bar(
            ["Raw sensor\nvalues", "Semantic\ntokens"],
            means, color=[WARN, OK], width=0.5,
        )
        ax.bar_label(bars, fmt="%.0f ms", color="white", fontsize=10, padding=4)
        ax.set_title("Mean Latency Comparison", color="white")
        ax.set_ylabel("ms", color="#94a3b8")
        speedup = means[0] / means[1] if means[1] > 0 else 1
        ax.text(
            0.5, 0.85, f"{speedup:.2f}× speedup with semantic tokens",
            transform=ax.transAxes, ha="center", color="#fbbf24", fontsize=10,
        )
        plt.tight_layout()
        self._savefig(fname)
