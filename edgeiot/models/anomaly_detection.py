import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from edgeiot.config import AnomalyConfig

logger = logging.getLogger(__name__)


class SequenceWindowGenerator:
    """Creates sliding-window sequences over sensor columns for time-series anomaly detection."""

    def __init__(self, sensor_cols: List[str], window_size: int = 5):
        self.sensor_cols = sensor_cols
        self.window_size = window_size

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build sliding windows of shape (n_samples - window_size, window_size * n_features).
        Each row is a flattened window of consecutive sensor readings.
        """
        n = len(df) - self.window_size
        if n <= 0:
            raise ValueError(
                f"DataFrame has {len(df)} rows, which is too few for "
                f"window_size={self.window_size}."
            )
        X_seq = np.array([
            df.iloc[i : i + self.window_size][self.sensor_cols].values.flatten()
            for i in range(n)
        ])
        logger.info(f"Generated {X_seq.shape[0]} sequences of width {X_seq.shape[1]}")
        return X_seq


class IsolationForestAnomalyDetector:
    """Trains an Isolation Forest on windowed sensor data and produces anomaly scores/labels."""

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.model = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=config.random_state,
        )
        self.scores: np.ndarray = np.array([])
        self.labels: np.ndarray = np.array([])

    def fit_predict(self, X_seq: np.ndarray) -> None:
        """Fit on sequences and store anomaly scores and string labels."""
        logger.info("Fitting Isolation Forest...")
        self.model.fit(X_seq)

        self.scores = self.model.decision_function(X_seq)
        raw_preds = self.model.predict(X_seq)
        self.labels = np.where(raw_preds == -1, "ABNORMAL", "NORMAL")

        n_anom = (self.labels == "ABNORMAL").sum()
        logger.info(
            f"Anomalies detected: {n_anom}/{len(self.labels)} "
            f"({n_anom / len(self.labels) * 100:.1f}%)"
        )

    @property
    def n_anomalies(self) -> int:
        return int((self.labels == "ABNORMAL").sum())
