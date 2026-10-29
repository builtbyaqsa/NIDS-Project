import os
from src.hyperparameter_tuning import NIDSModelTuner
from src.utils import setup_logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = setup_logger()

def main():
    logger.info("=== STARTING WEEK 10 HYPERPARAMETER TUNING PIPELINE ===")
    
    DATA_PATH = "data/processed/cleaned_nids_data.csv"
    FEATURES_PATH = "data/processed/selected_features.pkl"

    # 1. Initialize Tuner
    tuner = NIDSModelTuner(data_path=DATA_PATH, features_path=FEATURES_PATH)

    # 2. Load Data
    X_train, X_test, y_train, y_test = tuner.load_data()

    # 3. Perform Grid Search Tuning
    best_model = tuner.tune_random_forest(X_train, y_train)

    # 4. Evaluate Tuned Model on Test Set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n================ TUNED MODEL RESULTS ================")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)
    print("=====================================================\n")

    # 5. Save Tuned Model
    tuner.save_tuned_model(best_model)

    logger.info("=== WEEK 10 PIPELINE EXECUTED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()