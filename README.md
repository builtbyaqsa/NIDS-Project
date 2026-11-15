# Network Intrusion Detection System (NIDS)

AI-powered Network Intrusion Detection System built with Python, Streamlit, Scikit-learn, and SQLite for real-time threat detection, risk scoring, Explainable AI (XAI) insights, and SOC audit reporting.

## Features
- **Single Packet Analysis:** Interactive packet parameter inspection with threat status, confidence score, risk level, recommended SOC action, and XAI feature importance.
- **Batch CSV Upload Analytics:** Bulk network log evaluation with summary metrics, threat distribution chart, empty-file guard, and downloadable CSV results.
- **Live Threat Simulation:** Automated simulated packet stream logging alerts directly into SQLite.
- **Telemetry & Audit Logs:** Persistent SQLite logging (`logs/nids_telemetry.db`), prediction filtering, log refresh/clear controls, CSV log export, and downloadable Markdown incident reports (`.md`).

## Project Structure
```text
NIDS-Project/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   ├── risk_engine.py
│   ├── train_model.py
│   └── xai_explainer.py
├── data/
│   ├── raw/
│   └── processed/
│       └── tuned_model.pkl
└── logs/