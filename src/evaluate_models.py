import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLS = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes"]


def load_or_build_dataset(csv_path: str = "data/raw/sample_traffic.csv"):
    df = pd.read_csv(csv_path)
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if "label" in df.columns:
        y_raw = df["label"]
        if pd.api.types.is_numeric_dtype(y_raw):
            y = (pd.to_numeric(y_raw, errors="coerce").fillna(0) > 0).astype(int)
        else:
            y = (~y_raw.astype(str).str.lower().str.contains("normal")).astype(int)
    else:
        # Fallback synthetic target if sample CSV has no label column
        rng = np.random.default_rng(42)
        score = (X["src_bytes"] > X["src_bytes"].median()).astype(int) + (
            X["dst_bytes"] > X["dst_bytes"].median()
        ).astype(int)
        noise = rng.integers(0, 2, size=len(X))
        y = ((score + noise) >= 2).astype(int)

    return X, y


def evaluate_all_models():
    X, y = load_or_build_dataset()

    stratify_arg = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify_arg
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=42),
            X_train,
            X_test,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            X_train,
            X_test,
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_scaled,
            X_test_scaled,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            X_train,
            X_test,
        ),
    }

    print(f"\nDataset rows: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print("-" * 92)
    print(
        f"{'Model':<22} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'FPR':<8} {'Inf(ms)':<10}"
    )
    print("-" * 92)

    for name, (clf, x_tr, x_te) in models.items():
        clf.fit(x_tr, y_train)
        start = time.perf_counter()
        preds = clf.predict(x_te)
        elapsed_ms = (time.perf_counter() - start) * 1000

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        print(
            f"{name:<22} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f} {fpr:<8.4f} {elapsed_ms:<10.3f}"
        )


if __name__ == "__main__":
    evaluate_all_models()