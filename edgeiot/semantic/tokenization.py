import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Threshold map ────────────────────────────────────────────────────────────
# Each entry: (column_name, operator, threshold, token_name)
DEFAULT_TOKEN_RULES = [
    ("temperature",    ">",   1, "[TEMP_HIGH]"),
    ("temperature",    "<",  -1, "[TEMP_LOW]"),
    ("humidity",       ">",   1, "[HUMIDITY_HIGH]"),
    ("humidity",       "<",  -1, "[HUMIDITY_LOW]"),
    ("noise_db",       ">",   1, "[NOISE_SPIKE]"),
    ("motion_level",   ">",   1, "[MOTION_ACTIVE]"),
    ("light_intensity","<",  -1, "[LOW_LIGHT]"),
]


class SemanticTokenizer:
    """Maps normalised sensor deviations to domain-specific semantic tokens."""

    def __init__(self, rules: Optional[List[tuple]] = None):
        self.rules = rules or DEFAULT_TOKEN_RULES

    def tokenize_row(self, row: pd.Series) -> List[str]:
        """Apply rules to a single row and return matched tokens."""
        tokens: List[str] = []
        for col, op, threshold, token in self.rules:
            val = row.get(col, 0.0)
            if op == ">" and val > threshold:
                tokens.append(token)
            elif op == "<" and val < threshold:
                tokens.append(token)
        return tokens

    @staticmethod
    def add_anomaly_flag(tokens: List[str], anomaly_status: str) -> List[str]:
        """Append [ANOMALY_ALERT] if the sample was flagged abnormal."""
        if anomaly_status == "ABNORMAL":
            tokens.append("[ANOMALY_ALERT]")
        return tokens

    def tokenize_dataframe(
        self,
        df: pd.DataFrame,
        anomaly_labels: Optional[np.ndarray] = None,
    ) -> List[List[str]]:
        """Tokenize every row of a DataFrame, optionally merging anomaly flags."""
        all_tokens: List[List[str]] = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            tokens = self.tokenize_row(row)
            if anomaly_labels is not None and idx < len(anomaly_labels):
                tokens = self.add_anomaly_flag(tokens, anomaly_labels[idx])
            all_tokens.append(tokens)
        return all_tokens

    @staticmethod
    def token_distribution(token_lists: List[List[str]]) -> Dict[str, int]:
        """Return a frequency counter over all token lists."""
        flat = [t for tl in token_lists for t in tl]
        return dict(Counter(flat))


class TransformerEncoder:
    """Encodes semantic token streams with a Hugging Face tokenizer (offline)."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load so import time stays low on resource-constrained devices."""
        if self._tokenizer is None:
            logger.info(f"Loading HF tokenizer: {self.model_name}")
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def encode(self, text: str, max_length: int = 64) -> Dict:
        """Tokenize a text string and return token IDs + attention mask."""
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

    def encode_batch(self, texts: List[str], max_length: int = 64) -> Dict:
        """Tokenize a list of texts."""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )


# ── Output builder ───────────────────────────────────────────────────────────
def build_output_record(
    prediction: int,
    confidence: float,
    tokens: List[str],
) -> Dict:
    """Construct a single pipeline output record conforming to schema v2.0."""
    return {
        "schema_version": "2.0",
        "task_id":        "smart_home_edge_ai",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "prediction":     {"label": int(prediction), "confidence": float(round(confidence, 3))},
        "semantic_tokens": tokens,
    }
