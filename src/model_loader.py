import os
import joblib
import numpy as np

MODEL_PATH = "data/processed/tuned_model.pkl"

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def predict_traffic(model, feature_array):
    """
    Runs model inference if shape matches; falls back cleanly to heuristic
    if the saved model expects a different feature count than the UI inputs.
    """
    if model is not None:
        try:
            expected_features = getattr(model, "n_features_in_", None)
            if expected_features is None or expected_features == feature_array.shape[1]:
                pred = int(model.predict(feature_array)[0])
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feature_array)[0]
                    conf = float(np.max(proba))
                else:
                    conf = 0.90
                return pred, round(conf, 2), "Random Forest (tuned_model.pkl)"
        except Exception:
            pass

    # Fallback rule if feature schema differs
    src_bytes = feature_array[0][3]
    pred = 1 if src_bytes > 3000 else 0
    conf = 0.92 if pred == 1 else 0.95
    return pred, conf, "Fallback Heuristic"