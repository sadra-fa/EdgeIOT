#!/usr/bin/env python3
"""
leakage_check.py — How much of the classifiers' accuracy comes from the
one-hot `behaviour` column?

Uses the cluster labels of the pipeline run (re-derived here from the
`behaviour` column, which they match except for two rows, to avoid repeating
the expensive Ward clustering) and trains the same XGBoost configuration on
three feature sets: all 24 features, sensors + location, and the five sensor
readings only. Writes outputs/rerun/<run>/leakage_check.json.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgeiot.config import PipelineConfig                               # noqa: E402
from edgeiot.data import DataLoader, DataPreprocessor                    # noqa: E402

GROUP_OF_BEHAVIOUR = {"studying": 0, "eating": 0, "stressed": 1, "walking": 1, "sleeping": 2}


def main():
    cfg = PipelineConfig()
    run = sorted((PROJECT_ROOT / "outputs" / "rerun").glob("*"))[-1]
    df = DataLoader(str(PROJECT_ROOT / "RealLife.csv")).load_and_clean()
    X = DataPreprocessor(cfg.preprocessing).preprocess(df)
    y = df["behaviour"].map(GROUP_OF_BEHAVIOUR).values

    sensors = cfg.preprocessing.sensor_cols
    feature_sets = {
        "all_24_features": list(X.columns),
        "sensors_and_location": sensors + [c for c in X.columns if c.startswith("location_")],
        "sensors_only": sensors,
    }
    results = {}
    for name, cols in feature_sets.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X[cols], y, test_size=0.2, random_state=42, stratify=y)
        clf = XGBClassifier(**cfg.classification.xgboost_params)
        clf.fit(X_tr, y_tr)
        results[name] = {"features": len(cols), "accuracy": float((clf.predict(X_te) == y_te).mean())}
        print(name, results[name])
    (run / "leakage_check.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
