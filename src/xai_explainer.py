import joblib
import pandas as pd

class XAIExplainer:
    """
    Provides model interpretability by extracting feature importance 
    or local explanations for NIDS predictions.
    """
    def __init__(self, model_path: str = "data/processed/tuned_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the trained ML model if available."""
        try:
            self.model = joblib.load(self.model_path)
            print("[INFO] Model loaded successfully for XAI analysis.")
        except Exception as e:
            print(f"[WARNING] Could not load model yet: {e}")

    def explain_prediction(self, input_features: list, feature_names: list) -> dict:
        """
        Extracts feature contributions or top contributing factors for a given sample.
        """
        if not self.model or not hasattr(self.model, "feature_importances_"):
            return {"explanation": "Model or feature importances not available."}
        
        importances = self.model.feature_importances_
        feature_importance_map = sorted(
            zip(feature_names, importances), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Top 3 contributing features
        top_features = feature_importance_map[:3]
        
        return {
            "top_risk_factors": [{"feature": f, "importance_score": round(score, 4)} for f, score in top_features]
        }

if __name__ == "__main__":
    # Test XAI module locally
    explainer = XAIExplainer()
    print("XAI Module initialized successfully.")