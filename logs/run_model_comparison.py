import os
import pandas as pd
from src.model_comparison import NIDSModelComparator
from src.utils import setup_logger

logger = setup_logger()

def main():
    logger.info("=== STARTING WEEK 9 MODEL COMPARISON PIPELINE ===")
    
    DATA_PATH = "data/processed/cleaned_nids_data.csv"
    FEATURES_PATH = "data/processed/selected_features.pkl"

    # 1. Initialize Comparator
    comparator = NIDSModelComparator(data_path=DATA_PATH, features_path=FEATURES_PATH)

    # 2. Prepare Data
    X_train, X_test, y_train, y_test = comparator.prepare_data()

    # 3. Evaluate All Models
    comparison_df = comparator.evaluate_models(X_train, X_test, y_train, y_test)

    # Print Clean Comparison Table in Terminal
    print("\n================ MODEL COMPARISON RESULTS ================")
    print(comparison_df.to_string(index=False))
    print("==========================================================\n")

    # 4. Save comparison report
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "model_comparison_report.csv")
    comparison_df.to_csv(report_path, index=False)
    logger.info(f"Model comparison report saved to {report_path}")

    logger.info("=== WEEK 9 PIPELINE EXECUTED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()