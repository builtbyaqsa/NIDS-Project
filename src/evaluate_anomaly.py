import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from evaluate_models import FEATURE_COLS, load_or_build_dataset


def run_cv_and_anomaly_check(csv_path: str = "data/raw/sample_traffic.csv"):
    X, y = load_or_build_dataset(csv_path)

    min_class_count = int(y.value_counts().min())
    n_splits = min(5, max(2, min_class_count))

    print(f"\nRows: {len(X)} | Min class count: {min_class_count} | CV splits: {n_splits}")

    if min_class_count >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
        print(f"RF CV Accuracies: {np.round(scores, 4)} | Mean: {scores.mean():.4f}")
    else:
        print("CV skipped: fewer than 2 samples in a class for stratified split.")

    iso = IsolationForest(contamination=0.25, random_state=42)
    # -1 indicates anomaly, 1 indicates normal/inlier
    preds = iso.fit_predict(X)
    scores_normality = iso.decision_function(X)

    result_df = X[FEATURE_COLS].copy()
    result_df["anomaly_flag"] = np.where(preds == -1, "Anomaly", "Normal")
    result_df["anomaly_score"] = np.round(scores_normality, 4)

    print("\nAnomaly Detection Output (IsolationForest):")
    print(result_df.to_string(index=True))


if __name__ == "__main__":
    run_cv_and_anomaly_check()