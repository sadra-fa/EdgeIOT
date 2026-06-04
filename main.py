#!/usr/bin/env python3
"""
main.py — Smart Home Edge AI Pipeline Orchestrator
====================================================
Runs the full pipeline: load → preprocess → cluster → classify → anomaly detect
→ semantic tokenize → LLM reason → visualise → export.

Usage:
    python main.py                                    # defaults
    python main.py --data /path/to/RealLife.csv       # custom CSV
    python main.py --model phi3.5                     # different Ollama model
    python main.py --skip-llm                         # skip LLM benchmark
"""

import os, sys, json, argparse, logging, warnings, subprocess
from datetime import datetime

os.environ["TRANSFORMERS_NO_TF"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from edgeiot.config import PipelineConfig
from edgeiot.data import DataLoader, DataPreprocessor, HierarchicalClusteringManager
from edgeiot.models import ModelEvaluator, XGBoostWrapper, SequenceWindowGenerator, IsolationForestAnomalyDetector
from edgeiot.semantic import SemanticTokenizer, TransformerEncoder, build_output_record, OllamaReasoner, CLUSTER_NAMES, discover_ollama_models, select_model_interactive
from edgeiot.viz import PipelineVisualizer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True
)
logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Smart Home Edge AI Pipeline")
    p.add_argument("--data",     type=str, default=None,   help="Path to CSV dataset")
    p.add_argument("--model",    type=str, default=None,   help="Ollama model name (e.g. phi3.5)")
    p.add_argument("--skip-llm", action="store_true",      help="Skip LLM reasoning / benchmark")
    p.add_argument("--outdir",   type=str, default=None,   help="Output directory")
    return p.parse_args()

def main():
    args = parse_args()

    # ── Configuration ────────────────────────────────────────────────────
    cfg = PipelineConfig()
    if args.data:
        cfg.data_path = args.data
    
    llm_sample_count = None
    if not args.skip_llm and not args.model:
        models = discover_ollama_models()
        if not models:
            logger.warning("No Ollama models found or unreachable. Skipping LLM reasoning.")
            args.skip_llm = True
        else:
            cfg.ollama.model = select_model_interactive(models)
    elif args.model:
        cfg.ollama.model = args.model
        
    if not args.skip_llm:
        try:
            count_str = input("\nHow many samples to process with LLM? (Enter number or leave blank for all): ").strip()
            if count_str.isdigit():
                llm_sample_count = int(count_str)
        except Exception:
            pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = cfg.ollama.model.replace(":", "_").replace("/", "_")
    run_id = f"{timestamp}_{safe_model}"

    base_outdir = args.outdir if args.outdir else "outputs"
    cfg.output_dir = os.path.join(base_outdir, run_id)
    cfg.plots_dir  = os.path.join(cfg.output_dir, "plots")
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    for d in [
        "01_data_exploration", "02_clustering", "03_classification", 
        "04_anomaly_detection", "05_tokenization", "06_llm_benchmark"
    ]:
        os.makedirs(os.path.join(cfg.plots_dir, d), exist_ok=True)

    viz = PipelineVisualizer(output_dir=cfg.plots_dir)

    print("\n" + "═"*70)
    print(" 1. LOAD & CLEAN")
    print("═"*70)
    loader = DataLoader(cfg.data_path)
    df = loader.load_and_clean()

    print("\n" + "═"*70)
    print(" 2. ENCODE + NORMALIZE")
    print("═"*70)
    preprocessor = DataPreprocessor(cfg.preprocessing)
    X_processed = preprocessor.preprocess(df)

    print("\n" + "═"*70)
    print(" 3. SENSOR TIME-SERIES PLOT")
    print("═"*70)
    viz.plot_timeseries(X_processed, cfg.preprocessing.sensor_cols[:3])

    print("\n" + "═"*70)
    print(" 4. PCA + HIERARCHICAL CLUSTERING")
    print("═"*70)
    X_dense = X_processed.values
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_dense)

    cluster_mgr = HierarchicalClusteringManager(cfg.clustering)
    pseudo_labels = cluster_mgr.fit_predict(X_processed)

    logger.info(
        f"Cluster sizes: {dict(zip(*np.unique(pseudo_labels, return_counts=True)))}"
    )
    viz.plot_pca_clustering(X_pca_2d, pseudo_labels)

    print("\n" + "═"*70)
    print(" 5. TRAIN / TEST SPLIT")
    print("═"*70)
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X_processed, pseudo_labels
    )

    # ══════════════════════════════════════════════════════════════════════
    # 6-8. CLASSIFIERS (Decision Tree, Random Forest, XGBoost)
    # ══════════════════════════════════════════════════════════════════════
    evaluator = ModelEvaluator(cfg.classification)
    classifiers = evaluator.build_all_classifiers(
        num_class=cfg.clustering.n_clusters
    )

    for clf in classifiers:
        clf.fit(X_train, y_train)
        clf.evaluate(X_test, y_test)

    # Identify the XGBoost classifier for detailed analysis
    xgb_clf = next(c for c in classifiers if isinstance(c, XGBoostWrapper))
    y_pred_xgb   = xgb_clf.predict(X_test)
    y_proba      = xgb_clf.predict_proba(X_test)
    y_confidence = np.max(y_proba, axis=1)

    # Classification report
    report = evaluator.classification_report_str(y_test, y_pred_xgb)
    logger.info(f"\nXGBoost Classification Report:\n{report}")

    # Cross-validation
    cv_mean, cv_std = evaluator.cross_validate(
        xgb_clf.model, X_processed, pseudo_labels, cv=5
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    viz.plot_confusion_matrix(y_test, y_pred_xgb)
    viz.plot_feature_importance(xgb_clf.model)

    y_test_bin = evaluator.binarize_labels(y_test)
    viz.plot_multiclass_roc(xgb_clf.model, X_test, y_test_bin)

    viz.plot_model_comparison(
        [c.name for c in classifiers],
        [c.accuracy for c in classifiers],
    )

    print("\n" + "═"*70)
    print(" 9. SHAP")
    print("═"*70)
    try:
        shap_values, _ = evaluator.compute_shap_values(xgb_clf.model, X_test)
        viz.plot_shap_beeswarm(shap_values)
        logger.info("SHAP done.")
    except Exception as e:
        logger.warning(f"SHAP skipped: {e}")

    print("\n" + "═"*70)
    print(" 10. TIME-WINDOW ANOMALY DETECTION")
    print("═"*70)
    window_gen = SequenceWindowGenerator(
        cfg.preprocessing.sensor_cols, cfg.anomaly.window_size
    )
    X_seq = window_gen.transform(X_processed)

    anomaly_detector = IsolationForestAnomalyDetector(cfg.anomaly)
    anomaly_detector.fit_predict(X_seq)

    viz.plot_anomaly_scores(anomaly_detector.scores, anomaly_detector.labels)

    print("\n" + "═"*70)
    print(" 11. SEMANTIC TOKENIZATION")
    print("═"*70)
    tokenizer = SemanticTokenizer()

    outputs = []
    for i in range(len(X_test)):
        row    = X_test.iloc[i]
        tokens = tokenizer.tokenize_row(row)
        status = (
            anomaly_detector.labels[i]
            if i < len(anomaly_detector.labels)
            else "NORMAL"
        )
        tokens = tokenizer.add_anomaly_flag(tokens, status)
        outputs.append(
            build_output_record(y_pred_xgb[i], y_confidence[i], tokens)
        )

    # Token distribution
    token_counts = tokenizer.token_distribution(
        [o["semantic_tokens"] for o in outputs]
    )
    viz.plot_token_distribution(token_counts)

    print("\n" + "═"*70)
    print(" 12. LLM REASONING (optional)")
    print("═"*70)
    df_bench = None
    df_eff   = None
    n_anom   = anomaly_detector.n_anomalies

    if not args.skip_llm:
        reasoner = OllamaReasoner(cfg.ollama)

        # 12a. Smoke test
        logger.info("Smoke test — first 3 samples:")
        reasoner.smoke_test(outputs, n=3)

        # 12b. Full benchmark
        records = reasoner.benchmark(outputs, limit=llm_sample_count)
        df_bench = pd.DataFrame(records)
        df_bench.to_csv(f"{cfg.output_dir}/jetson_benchmark.csv", index=False)
        logger.info(f"Benchmark saved → {cfg.output_dir}/jetson_benchmark.csv")
        logger.info(
            f"\nSummary statistics:\n"
            f"{df_bench[['latency_ms','tokens_out','tokens_per_sec','ram_mb']].describe().round(2)}"
        )

        viz.plot_llm_benchmark(
            df_bench,
            suptitle=f"{cfg.ollama.model} — Jetson Orin Nano Benchmark",
        )

        # Attach reasoning back to outputs
        for i, rec in enumerate(records):
            outputs[i]["llm_reasoning"] = rec["reasoning"]

        # 12c. Token efficiency experiment
        logger.info("Running token efficiency experiment (20 samples)...")
        efficiency_records = []
        sample_rows = X_test.iloc[:20]
        for i in range(len(sample_rows)):
            row   = sample_rows.iloc[i]
            label = y_pred_xgb[i]
            raw_prompt_tokens = [
                f"temperature={row['temperature']:.3f}",
                f"humidity={row['humidity']:.3f}",
                f"noise_db={row['noise_db']:.3f}",
                f"light_intensity={row['light_intensity']:.3f}",
                f"motion_level={row['motion_level']:.3f}",
            ]
            sem_tokens = tokenizer.tokenize_row(row) or ["[NO_EVENT]"]
            r_raw = reasoner.reason(raw_prompt_tokens, label)
            r_sem = reasoner.reason(sem_tokens, label)
            efficiency_records.append({
                "sample":         i,
                "raw_input_len":  len(" ".join(raw_prompt_tokens)),
                "sem_input_len":  len(" ".join(sem_tokens)),
                "raw_latency_ms": r_raw["latency_ms"],
                "sem_latency_ms": r_sem["latency_ms"],
                "raw_tps":        r_raw["tokens_per_sec"],
                "sem_tps":        r_sem["tokens_per_sec"],
            })

        df_eff = pd.DataFrame(efficiency_records)
        df_eff.to_csv(f"{cfg.output_dir}/token_efficiency.csv", index=False)
        viz.plot_token_efficiency(df_eff)

        speedup = (
            df_eff["raw_latency_ms"].mean() / df_eff["sem_latency_ms"].mean()
            if df_eff["sem_latency_ms"].mean() > 0
            else 1
        )
        logger.info(f"  Raw input  avg latency : {df_eff['raw_latency_ms'].mean():.1f} ms")
        logger.info(f"  Semantic   avg latency : {df_eff['sem_latency_ms'].mean():.1f} ms")
        logger.info(f"  Speedup                : {speedup:.2f}×")
    else:
        logger.info("LLM reasoning skipped (--skip-llm).")

    print("\n" + "═"*70)
    print(" 13. COMPILE FINAL JSON OUTPUT")
    print("═"*70)
    with open(f"{cfg.output_dir}/semantic_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)

    with open(f"{cfg.output_dir}/final_outputs_with_reasoning.json", "w") as f:
        json.dump(outputs[:10], f, indent=2)

    logger.info("\nFINAL OUTPUT EXAMPLE (with LLM reasoning):")
    logger.info(json.dumps(outputs[0], indent=2))

    print("\n" + "═"*70)
    print(" 14. DISSERTATION SUMMARY")
    print("═"*70)
    dt_acc = classifiers[0].accuracy
    rf_acc = classifiers[1].accuracy
    xgb_acc = classifiers[2].accuracy

    summary = f"""
╔══════════════════════════════════════════════════════════╗
║     SMART HOME EDGE AI — DISSERTATION RESULTS           ║
╠══════════════════════════════════════════════════════════╣
║  CLASSIFICATION                                         ║
║    Decision Tree    : {dt_acc:.4f}                          ║
║    Random Forest    : {rf_acc:.4f}                          ║
║    XGBoost          : {xgb_acc:.4f}                          ║
║    XGBoost CV 5-fold: {cv_mean:.3f} ± {cv_std:.3f}                   ║
╠══════════════════════════════════════════════════════════╣
║  ANOMALY DETECTION                                      ║
║    Flagged          : {n_anom}/{len(anomaly_detector.labels)} ({n_anom/len(anomaly_detector.labels)*100:.1f}%)                    ║
╠══════════════════════════════════════════════════════════╣"""

    if df_bench is not None:
        avg_lat = df_bench["latency_ms"].mean()
        p95_lat = df_bench["latency_ms"].quantile(0.95)
        avg_tps = df_bench["tokens_per_sec"].mean()
        avg_ram = df_bench["ram_mb"].mean()
        summary += f"""
║  {cfg.ollama.model.upper()} ON JETSON ORIN NANO             ║
║    Avg latency      : {avg_lat:.1f} ms                        ║
║    P95 latency      : {p95_lat:.1f} ms                        ║
║    Avg tokens/sec   : {avg_tps:.1f}                           ║
║    Avg RAM usage    : {avg_ram:.1f} MB                       ║"""

    if df_eff is not None:
        summary += f"""
╠══════════════════════════════════════════════════════════╣
║  TOKEN EFFICIENCY                                       ║
║    Raw input latency: {df_eff['raw_latency_ms'].mean():.1f} ms                       ║
║    Semantic latency : {df_eff['sem_latency_ms'].mean():.1f} ms                       ║
║    Speedup          : {speedup:.2f}×                            ║"""

    summary += """
╚══════════════════════════════════════════════════════════╝"""

    print(summary)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
