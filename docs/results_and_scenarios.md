# Controlled Scenarios & Research Results Template

## 1. Controlled Evaluation Scenarios
| Scenario ID | Scenario Name | Input Source | Expected Behavior | Logged Output |
| :--- | :--- | :--- | :--- | :--- |
| S1 | Normal Traffic | Single Packet / CSV | Low risk, Normal class | SQLite row + XAI top factors |
| S2 | Known Attack | Single Packet / CSV | Elevated risk, Attack class | SQLite row + Recommended action |
| S3 | Batch Traffic Mix | `data/raw/sample_traffic.csv` | Per-row predictions + chart | `nids_batch_results.csv` |
| S4 | Live Stream Burst | Live Threat Simulation | Continuous streaming & logging | SQLite telemetry entries |
| S5 | Unusual / Novel Flow | Edge-case feature values | Risk + XAI + Anomaly inspection | Audit CSV + `.md` Incident Report |

## 2. Summary Tables & Figures Checklist
- **Table 1:** Model comparison (`tuned_model.pkl` baseline vs candidate models)
- **Table 2:** Risk-tier distribution (`Low`, `Medium`, `High`, `Critical`) and recommended SOC responses
- **Table 3:** Known vs anomalous traffic observations
- **Figures:** Confusion matrix, attack/risk distributions, XAI factor view, system architecture, and Streamlit dashboard screenshots