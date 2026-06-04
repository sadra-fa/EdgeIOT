import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from edgeiot.config import PreprocessingConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles dataset loading, cleaning, and basic preprocessing structure."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None

    def load_and_clean(self) -> pd.DataFrame:
        """Loads dataset from CSV, removes duplicates, and fills missing numeric values."""
        logger.info(f"Loading dataset from: {self.file_path}")
        try:
            self.raw_df = pd.read_csv(self.file_path)
            logger.info(f"Original Dataset Shape: {self.raw_df.shape}")
        except FileNotFoundError:
            logger.error(f"File not found at {self.file_path}. Please check the path.")
            raise

        # Drop duplicates
        self.cleaned_df = self.raw_df.drop_duplicates()
        
        # Fill numeric NaNs with column means
        numeric_cols = self.cleaned_df.select_dtypes(include="number").columns
        self.cleaned_df[numeric_cols] = self.cleaned_df[numeric_cols].fillna(
            self.cleaned_df[numeric_cols].mean()
        )
        
        logger.info(f"Cleaned Dataset Shape: {self.cleaned_df.shape}")
        return self.cleaned_df


class DataPreprocessor:
    """Manages scaling, categorical encoding, and splitting of time-series/sensor data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies One-Hot Encoding and Standard Scaling to numeric columns."""
        logger.info("Applying one-hot encoding to categorical features...")
        # Get dummies for categorical columns
        processed_df = pd.get_dummies(
            df,
            columns=self.config.categorical_cols
        )
        
        # Fit-transform standard scaler on sensor columns
        logger.info(f"Normalizing sensor features: {self.config.sensor_cols}")
        processed_df[self.config.sensor_cols] = self.scaler.fit_transform(
            processed_df[self.config.sensor_cols]
        )
        
        self.feature_names = processed_df.columns.tolist()
        logger.info(f"Processed Dataset Shape: {processed_df.shape}")
        return processed_df

    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series or np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series or np.ndarray, pd.Series or np.ndarray]:
        """Splits the preprocessed features and labels into train/test sets."""
        logger.info(f"Splitting dataset with test_size={self.config.test_size}")
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
