import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import shap

from edgeiot.config import ClassifierConfig

logger = logging.getLogger(__name__)


# ── Abstract base class ──────────────────────────────────────────────────────
class BaseClassifier(ABC):
    """Common interface for all classifiers used in the pipeline."""

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.accuracy: Optional[float] = None

    @abstractmethod
    def build_model(self) -> Any:
        """Return an instantiated sklearn/xgboost estimator."""

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        self.model = self.build_model()
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> float:
        """Evaluate on test set, store and return accuracy."""
        y_pred = self.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"{self.name} accuracy: {self.accuracy:.4f}")
        return self.accuracy


# ── Concrete classifiers ─────────────────────────────────────────────────────
class DecisionTreeWrapper(BaseClassifier):
    def __init__(self, params: Dict[str, Any]):
        super().__init__("Decision Tree", params.get("random_state", 42))
        self.params = params

    def build_model(self):
        return DecisionTreeClassifier(**self.params)


class RandomForestWrapper(BaseClassifier):
    def __init__(self, params: Dict[str, Any]):
        super().__init__("Random Forest", params.get("random_state", 42))
        self.params = params

    def build_model(self):
        return RandomForestClassifier(**self.params)


class XGBoostWrapper(BaseClassifier):
    def __init__(self, params: Dict[str, Any], num_class: int = 3):
        super().__init__("XGBoost", params.get("random_state", 42))
        self.params = {**params, "num_class": num_class, "verbosity": 0}

    def build_model(self):
        return XGBClassifier(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        super().fit(X_train, y_train)
        # Ensure feature names are stored on the booster for plot_importance
        self.model.get_booster().feature_names = list(X_train.columns)


# ── Evaluator ────────────────────────────────────────────────────────────────
class ModelEvaluator:
    """Centralises evaluation, cross-validation, and SHAP explainability."""

    def __init__(self, config: ClassifierConfig):
        self.config = config

    @staticmethod
    def classification_report_str(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        return classification_report(y_true, y_pred)

    @staticmethod
    def cross_validate(model, X: pd.DataFrame, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
        """Run k-fold cross-validation and return (mean, std) accuracy."""
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        logger.info(f"CV {cv}-fold: {scores.mean():.3f} ± {scores.std():.3f}")
        return float(scores.mean()), float(scores.std())

    @staticmethod
    def binarize_labels(y: np.ndarray) -> np.ndarray:
        """Binarize labels for multiclass ROC computation."""
        classes = np.unique(y)
        return label_binarize(y, classes=classes)

    @staticmethod
    def compute_shap_values(model, X_test: pd.DataFrame):
        """Compute SHAP values.  Returns (shap.Explanation, Explainer) or raises."""
        logger.info("Computing SHAP values...")
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        return shap_values, explainer

    def build_all_classifiers(self, num_class: int = 3) -> List[BaseClassifier]:
        """Instantiate the three standard classifiers from config."""
        return [
            DecisionTreeWrapper(self.config.decision_tree_params),
            RandomForestWrapper(self.config.random_forest_params),
            XGBoostWrapper(self.config.xgboost_params, num_class=num_class),
        ]
