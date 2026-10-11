import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from src.utils import setup_logger

logger = setup_logger()

class NIDSFeatureEngineer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None

    def load_processed_data(self) -> pd.DataFrame:
        """Loads the cleaned dataset generated from Week 6 preprocessing."""
        if not os.path.exists(self.data_path):
            logger.error(f"Processed dataset not found at {self.data_path}")
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
        logger.info(f"Loading processed data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df

    def analyze_correlation(self, threshold: float = 0.85) -> List[str]:
        """
        Calculates Pearson correlation matrix and identifies highly correlated 
        features to remove redundancy.
        """
        logger.info("Performing Correlation Analysis...")
        # Drop target column for correlation
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        corr_matrix = numeric_df.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        logger.info(f"Identified {len(to_drop)} highly correlated features to drop: {to_drop}")
        return to_drop

    def compute_feature_importance(self, target_col: str = 'label') -> pd.DataFrame:
        """
        Computes feature importance using a quick Random Forest Classifier.
        """
        logger.info("Computing Feature Importance using Random Forest...")
        from sklearn.ensemble import RandomForestClassifier

        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        # Train a quick lightweight RF model
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        logger.info("Feature importance computed successfully.")
        return importance_df

    def plot_and_save_visualizations(self, importance_df: pd.DataFrame, output_dir: str = "data/processed/"):
        """Generates and saves feature importance and correlation plots."""
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Generating feature engineering visualizations...")

        # 1. Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
        plt.title('Top 10 Most Important Features for Intrusion Detection')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {plot_path}")