# Dataset Investigation & Preprocessing Report

## 1. Dataset Overview
- **Active Sample File:** `data/raw/sample_traffic.csv`
- **Trained Model Artifact:** `data/processed/tuned_model.pkl` (`RandomForestClassifier`, 5 input features)
- **Feature Schema:** `duration`, `protocol_type`, `service`, `src_bytes`, `dst_bytes`

## 2. Data Quality & Exploratory Checks
- **Schema Consistency:** Verify column names, data types, and feature ordering (`duration`, `protocol_type`, `service`, `src_bytes`, `dst_bytes`) match model expectations.
- **Missing & Duplicate Records:** Check for `NaN`/null entries and exact duplicate rows prior to training or batch evaluation.
- **Class Distribution & Imbalance:** Inspect `Normal` vs `Attack` ratios and high-byte flow outliers (`src_bytes`, `dst_bytes`).

## 3. Preprocessing & Feature Justification (`src/preprocessing.py`)
- **Feature Alignment:** Ensures incoming single-packet inputs, CSV uploads, and live simulation rows map to the exact 5 expected features.
- **Type Coercion & Empty Guard:** Converts inputs to numeric `float` arrays and blocks empty CSV uploads in `app.py`.
- **Behavioral Indicators:** `src_bytes`, `dst_bytes`, `duration`, `protocol_type`, and `service` capture flow volume, protocol context, and connection duration relevant to intrusion triage.