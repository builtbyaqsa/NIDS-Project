# Controlled Scenarios & Research Results Template

## 1. Controlled Evaluation Scenarios
| Scenario ID | Scenario Name | Input Source | Expected Behavior | Logged Output |
| :--- | :--- | :--- | :--- | :--- |
| S1 | Normal Traffic | Single Packet / CSV | Low risk, Normal class | SQLite row + XAI top factors |
| S2 | Known Attack | Single Packet / CSV | Elevated risk, Attack class | SQLite row + Recommended action |
| S3 | Batch Traffic Mix | `data/raw/sample_traffic.csv` | Per-row predictions + chart | `nids_batch_results.csv` |
| S4 | Live Stream Burst | Live Threat Simulation | Continuous streaming & logging | SQLite telemetry entries |
| S5 | Unusual / Novel Flow | Edge-case feature values | Risk + XAI + Anomaly inspection | Audit CSV + `.md` Incident Report |

## 2. Cross-Validation & Anomaly Detection Observations (`src/evaluate_anomaly.py`)
- **4-Fold Stratified Cross-Validation (Random Forest):** Fold accuracies `[1.0, 0.5, 0.5, 0.5]` with mean `0.6250` across `8` sample rows (`2` validation rows per fold), highlighting high variance on small datasets.
- **Isolation Forest Anomaly Layer:** Flagged rows `3` (`src_bytes=7800, dst_bytes=50`, score `-0.0129`) and `4` (`duration=15, dst_bytes=1200`, score `-0.0608`) as anomalies while marking higher-normality rows (`0, 2, 5, 6, 7`) as normal (`0.0043` to `0.0894`).

## 3. Summary Tables & Figures Checklist
- **Table 1:** Model comparison (`tuned_model.pkl` baseline vs candidate models — recorded in `docs/model_evaluation_plan.md`)
- **Table 2:** Risk-tier distribution (`Low`, `Medium`, `High`, `Critical`) and recommended SOC responses
- **Table 3:** Known vs anomalous traffic observations (`IsolationForest` scores + `S1–S5` scenarios)
- **Figures:** Confusion matrix, attack/risk distributions, XAI factor view, system architecture, and Streamlit dashboard screenshots