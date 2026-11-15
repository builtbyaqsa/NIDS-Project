# Research Foundation — ML-Based Intelligent Network Intrusion Detection & Risk Analysis System

## 1. Problem Statement & Limitations of Conventional NIDS
- Traditional signature-based NIDS struggle against novel or polymorphic network attacks.
- Standard ML-based NIDS prototypes often output raw binary predictions (`0`/`1`) without risk prioritization, analyst explanations, or audit-ready SOC reporting.
- Supervised classifiers alone can miss anomalous traffic that falls outside trained attack classes.

## 2. Research Questions
- **RQ1 (Detection):** Can a machine learning classifier accurately distinguish normal traffic from intrusion traffic across network flow features?
- **RQ2 (Risk Prioritization):** How can model confidence and traffic indicators be mapped into actionable risk levels (`Low`, `Medium`, `High`, `Critical`) and SOC response actions?
- **RQ3 (Explainability):** How can Explainable AI (XAI) feature attributions clarify why a packet or batch flow was flagged?
- **RQ4 (Anomalous Behaviour):** How does an anomaly-oriented layer behave when evaluating unusual or unseen traffic patterns?

## 3. Research Gaps & Proposed Contributions
- **Gap 1 — Raw Alerts Without Prioritization:** Addressed via a structured Risk Engine (`src/risk_engine.py`) producing risk scores (`0–100`), risk tiers, and recommended SOC actions.
- **Gap 2 — Opaque Model Decisions:** Addressed via XAI feature importance reporting (`src/xai_explainer.py`) integrated into the dashboard.
- **Gap 3 — Limited Operational Visibility:** Addressed via SQLite telemetry (`logs/nids_telemetry.db`), CSV exports, and downloadable Markdown incident reports (`app.py`).
- **Gap 4 — Unseen / Anomalous Traffic:** Addressed via controlled scenario testing and anomaly/unusual-traffic evaluation.

## 4. Literature Comparison Matrix
| Study Focus | Typical Method | Common Dataset | Key Strength | Key Limitation |
| :--- | :--- | :--- | :--- | :--- |
| Supervised ML NIDS | Random Forest / Decision Tree / Ensemble | NSL-KDD / CICIDS | High tabular classification performance | False positives on distribution shift |
| Deep Learning NIDS | CNN / LSTM / Autoencoder | CICIDS2017 / UNSW-NB15 | Captures complex sequential patterns | High compute cost, low transparency |
| Explainable NIDS | SHAP / Feature Attribution | CICIDS / NSL-KDD | Feature-level interpretability | Extra overhead and alert noise |
| Risk & Anomaly NIDS | Scoring Engine + Anomaly Detector | Benchmark / Simulated Flows | Actionable triage & novel pattern view | Threshold sensitivity |

## 5. System Architecture
```text
Network Traffic (Single Packet / Batch CSV / Live Simulation)
        │
        ▼
Preprocessing & Feature Alignment (`src/preprocessing.py`)
        │
        ▼
ML Classifier (`src/model_loader.py` -> `data/processed/tuned_model.pkl`)
        │
        ├──► Risk Engine (`src/risk_engine.py` -> Score, Level, Action)
        ├──► Explainable AI (`src/xai_explainer.py` -> Top Risk Factors)
        └──► Anomaly / Unusual Traffic Inspection
                │
                ▼
SQLite Telemetry (`src/database.py` -> `logs/nids_telemetry.db`)
                │
                ▼
Streamlit Dashboard (`app.py` -> Metrics, Charts, CSV & `.md` Report)