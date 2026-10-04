import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import setup_logger

logger = setup_logger()

class NIDSDataPreprocessor:
    def __init__(self, categorical_cols: List[str] = None, target_col: str = 'label'):
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []

    def load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            logger.error(f"Dataset path not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading raw dataset from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    def clean_duplicates_and_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        df = df.drop_duplicates()
        dedup_count = len(df)
        logger.info(f"Removed {initial_count - dedup_count} duplicate rows.")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            logger.warning(f"Found {null_counts} missing/NaN values. Imputing numeric values with median.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df.dropna(inplace=True)
            logger.info("Missing values handled.")
        else:
            logger.info("Zero missing values detected.")

        return df

    def encode_and_scale(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        if is_training:
            y_encoded = self.label_encoder.fit_transform(y)
            logger.info(f"Target classes encoded: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        else:
            y_encoded = self.label_encoder.transform(y)

        if self.categorical_cols:
            logger.info(f"Applying One-Hot Encoding on: {self.categorical_cols}")
            X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)

        if is_training:
            self.feature_columns = X.columns.tolist()
        else:
            X = X.reindex(columns=self.feature_columns, fill_value=0)

        logger.info("Normalizing numerical features via StandardScaler...")
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        X_transformed = pd.DataFrame(X_scaled, columns=self.feature_columns)
        y_transformed = pd.Series(y_encoded, name=self.target_col)

        return X_transformed, y_transformed

    def save_artifacts(self, artifact_dir: str = "data/processed/"):
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(artifact_dir, "scaler.pkl"))
        joblib.dump(self.label_encoder, os.path.join(artifact_dir, "label_encoder.pkl"))
        joblib.dump(self.feature_columns, os.path.join(artifact_dir, "feature_columns.pkl"))
        logger.info(f"Preprocessing artifacts successfully saved to '{artifact_dir}'.")