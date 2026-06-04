import logging
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import issparse
from edgeiot.config import ClusteringConfig

logger = logging.getLogger(__name__)

class HierarchicalClusteringManager:
    """Manages agglomerative hierarchical clustering on preprocessed sensor features."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.model = AgglomerativeClustering(
            n_clusters=self.config.n_clusters,
            linkage=self.config.linkage
        )

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Fits hierarchical clustering and returns cluster labels."""
        logger.info(f"Running Hierarchical Clustering (Ward) with {self.config.n_clusters} clusters...")
        # Ensure we have a dense array for clustering if it is sparse
        dense_data = df.toarray() if issparse(df) else df
        
        labels = self.model.fit_predict(dense_data)
        logger.info("Hierarchical Clustering completed.")
        return labels

    def save_labeled_data(self, df: pd.DataFrame, labels: np.ndarray) -> None:
        """Saves a copy of the dataframe with appended cluster labels to a CSV file."""
        labeled_df = df.copy()
        labeled_df["label"] = labels
        
        logger.info(f"Saving labeled dataset to: {self.config.output_labeled_csv}")
        labeled_df.to_csv(self.config.output_labeled_csv, index=False)
