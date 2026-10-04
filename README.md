# 🛡️ Risk-Aware & Explainable Network Intrusion Detection System (NIDS)

A research-grade Network Intrusion Detection System (NIDS) designed to classify network traffic, evaluate attack risk scores, provide explainable AI (XAI) insights, and detect novel anomalies.

---

## 📌 Project Overview

Traditional NIDS solutions often suffer from high false-positive rates and lack interpretability. This project addresses these research gaps by integrating:
- **ML & Anomaly Detection:** Classifying known network attacks and detecting novel/unseen threat behaviors.
- **Dynamic Risk Engine:** Calculating contextual risk scores (0–100) to prioritize critical security alerts.
- **Explainable AI (XAI):** Utilizing SHAP/LIME to explain why a specific flow was flagged as malicious.
- **Interactive SOC Dashboard:** Streamlit/React-based enterprise monitoring interface.

---

## 📂 Repository Structure

```text
nids-project/
│
├── data/
│   ├── raw/          # Raw network traffic datasets (e.g., NSL-KDD / CICIDS)
│   └── processed/    # Preprocessed, scaled, and encoded datasets
│
├── src/
│   ├── utils.py               # Custom logger and helper utilities
│   └── preprocessing.py       # Modular data cleaning & scaling engine
│
├── logs/             # Automated execution history logs
├── run_preprocessing.py # Pipeline entry point for data preprocessing
├── requirements.txt  # Python environment dependencies
└── README.md