# Model Evaluation & Experimentation Plan

## 1. Baseline & Model Comparison Matrix
Compare candidate models across standard intrusion detection metrics and runtime cost:

| Model | Accuracy | Precision | Recall | F1-Score | False Positive Rate | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (`tuned_model.pkl` — Random Forest) | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Gradient Boosting / SVM | TBD | TBD | TBD | TBD | TBD | TBD |

## 2. Error & Robustness Analysis
- **Confusion Matrix & Class Breakdown:** Inspect false positives (`Normal` flagged as `Attack`) and false negatives (`Attack` missed as `Normal`).
- **Cross-Validation & Generalization:** Evaluate performance across multiple splits to check overfitting on training flows.
- **Unseen / Anomalous Traffic:** Test behavior on traffic patterns outside known training classes and log limitations honestly.

## 3. Integrated Decision Pipeline (`app.py` + `src/`)
- **ML Detection & Classification:** `src/model_loader.py` (`data/processed/tuned_model.pkl`)
- **Risk Prioritization:** `src/risk_engine.py` (Risk score `0–100`, risk tier, recommended SOC response)
- **Explainable AI (XAI):** `src/xai_explainer.py` (Top contributing flow features)
- **SOC Telemetry & Audit:** `src/database.py` (`logs/nids_telemetry.db`), CSV export, and Markdown incident report