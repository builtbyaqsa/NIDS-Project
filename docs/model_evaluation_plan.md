\# Model Evaluation & Experimentation Plan

## 1. Baseline & Model Comparison Matrix
Evaluated on `data/raw/sample_traffic.csv` (`8` total rows: `5` train, `3` test; synthetic fallback labels where ground-truth labels were absent):

| Model | Accuracy | Precision | Recall | F1-Score | False Positive Rate | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (`tuned_model.pkl` / Random Forest) | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 1.0000 | 19.575 ms |
| Decision Tree | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 1.0000 | 1.217 ms |
| Logistic Regression | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 1.0000 | 0.422 ms |
| Gradient Boosting | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 1.0000 | 2.135 ms |

## 2. Error & Robustness Analysis
- **Small Sample & High FPR Finding:** With only `3` test samples (`5` train rows), all four models produced `1.0000` FPR and `0.3333` accuracy, showing that tiny sample sizes distort evaluation metrics and cause false alarms.
- **Inference Speed vs Model Complexity:** `Logistic Regression` (`0.422 ms`) and `Decision Tree` (`1.217 ms`) had lower test-batch latency than `Gradient Boosting` (`2.135 ms`) and `Random Forest` (`19.575 ms`).
- **Next Evaluation Step:** Benchmark on a larger multi-class network intrusion dataset (e.g., NSL-KDD / CICIDS subset) with stratified cross-validation and confusion matrices.

## 3. Integrated Decision Pipeline (`app.py` + `src/`)
- **ML Detection & Classification:** `src/model_loader.py` (`data/processed/tuned_model.pkl`)
- **Risk Prioritization:** `src/risk_engine.py` (Risk score `0–100`, risk tier, recommended SOC response)
- **Explainable AI (XAI):** `src/xai_explainer.py` (Top contributing flow features)
- **SOC Telemetry & Audit:** `src/database.py` (`logs/nids_telemetry.db`), CSV export, and Markdown incident report