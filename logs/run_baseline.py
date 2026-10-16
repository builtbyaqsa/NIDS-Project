import os
from src.baseline_model import NIDSBaselineModel
from src.utils import setup_logger

logger = setup_logger()

def main():
    logger.info("=== STARTING WEEK 8 BASELINE MODEL PIPELINE ===")
    
    DATA_PATH = "data/processed/cleaned_nids_data.csv"
    FEATURES_PATH = "data/processed/selected_features.pkl"

    # 1. Initialize Baseline Pipeline Class
    baseline = NIDSBaselineModel(data_path=DATA_PATH, features_path=FEATURES_PATH)

    # 2. Load Data and Split into Train/Test
    X_train, X_test, y_train, y_test = baseline.load_data_and_features()

    # 3. Train Model and Get Evaluation Metrics
    accuracy, report, conf_matrix = baseline.train_and_evaluate(X_train, X_test, y_train, y_test)

    # Print Clean Results in Terminal
    print("\n================ BASELINE MODEL RESULTS ================")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)
    print("========================================================\n")

    # 4. Save Model Artifact
    baseline.save_model()

    logger.info("=== WEEK 8 BASELINE PIPELINE EXECUTED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()