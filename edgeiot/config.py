import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class PreprocessingConfig:
    sensor_cols: List[str] = field(default_factory=lambda: [
        "temperature",
        "humidity",
        "noise_db",
        "light_intensity",
        "motion_level"
    ])
    categorical_cols: List[str] = field(default_factory=lambda: [
        "location",
        "behaviour"
    ])
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ClusteringConfig:
    n_clusters: int = 3
    linkage: str = "ward"
    output_labeled_csv: str = "dataset_with_labels_Hierarchical.csv"

@dataclass
class ClassifierConfig:
    random_state: int = 42
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42
    })
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "random_state": 42
    })
    decision_tree_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_depth": 5,
        "random_state": 42
    })

@dataclass
class AnomalyConfig:
    window_size: int = 5
    contamination: float = 0.05
    n_estimators: int = 100
    random_state: int = 42

@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434/api/generate"
    model: str = "llama3.2:3b"  # Can be parameterized (llama3.2:3b, phi3.5, etc.)

@dataclass
class PipelineConfig:
    data_path: str = "RealLife.csv"
    output_dir: str = "outputs"
    plots_dir: str = "outputs/plots"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    classification: ClassifierConfig = field(default_factory=ClassifierConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    transformer_model: str = "distilbert-base-uncased"
    
    def __post_init__(self):
        # Resolve data path if default and not present locally, but present in dissertation
        if self.data_path == "RealLife.csv" and not os.path.exists("RealLife.csv"):
            dissertation_data = os.path.join("..", "dissertation", "RealLife.csv")
            if os.path.exists(dissertation_data):
                self.data_path = dissertation_data
            else:
                # Also check absolute path
                abs_dissertation_data = "/home/sadra/dev/dissertation/RealLife.csv"
                if os.path.exists(abs_dissertation_data):
                    self.data_path = abs_dissertation_data
                    
        pass
