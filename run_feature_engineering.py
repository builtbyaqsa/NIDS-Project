import os
import pandas as pd
from src.feature_engineering import NIDSFeatureEngineer
from src.utils import setup_logger

logger = setup_logger()

def main():
    logger.info("=== STARTING WEEK 7 FEATURE ENGINEERING & SELECTION PIPELINE ===")
    
    PROCESSED_DATA_PATH = "data/processed/cleaned_nids_data.csv"
    
    # 1. Initialize Feature Engineer
    engineer = NIDSFeatureEngineer(data_path=PROCESSED_DATA_PATH)
    
    # 2. Load Data
    df = engineer.load_processed_data()
    
    # 3. Correlation Analysis
    dropped_features = engineer.analyze_correlation(threshold=0.85)
    
    # 4. Feature Importance Calculation
    importance_df = engineer.compute_feature_importance(target_col='label')
    
    print("\n--- TOP 10 IMPORTANT FEATURES ---")
    print(importance_df.head(10))
    print("------------------------------------\n")

    # 5. Save Visualizations & Selected Features Report
    engineer.plot_and_save_visualizations(importance_df)
    
    # Select top features (e.g., importance > 0.01 or top N features)
    selected_features = importance_df[importance_df['Importance'] > 0.005]['Feature'].tolist()
    
    # Save selected feature names to a file for Week 8 baseline model
    import joblib
    joblib.dump(selected_features, "data/processed/selected_features.pkl")
    logger.info(f"Selected {len(selected_features)} features saved to 'data/processed/selected_features.pkl'.")
    
    logger.info("=== WEEK 7 PIPELINE EXECUTED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()