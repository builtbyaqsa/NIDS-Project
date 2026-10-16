import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from src.utils import setup_logger

logger = setup_logger()

class NIDSModelComparator:
    """
    Trains and compares multiple machine learning models for Network Intrusion Detection
    to determine the optimal algorithm based on performance metrics.
    """
    def __init__(self, data_path: str, features_path: str):
        self.data_path = data_path
        self.features_path = features_path
        
        # Define candidate models for comparison
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

    def prepare_data(self):
        """Loads data, applies selected features from Week 7, and splits into train/test."""
        logger.info("Loading data for model comparison...")
        df = pd.read_csv(self.data_path)
        
        if os.path.exists(self.features_path):
            selected_features = joblib.load(self.features_path)
            if 'label' not in selected_features:
                selected_features.append('label')
            df = df[selected_features]
            logger.info(f"Loaded {len(selected_features)-1} selected features for comparison.")
        
        X = df.drop(columns=['label'])
        y = df['label']
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def evaluate_models(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        """Trains each model and computes key evaluation metrics."""
        results = []
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                "Model": name,
                "Accuracy": round(acc * 100, 2),
                "Precision": round(prec * 100, 2),
                "Recall": round(rec * 100, 2),
                "F1-Score": round(f1 * 100, 2)
            })
            logger.info(f"{name} evaluated successfully. Accuracy: {acc*100:.2f}%")
            
        return pd.DataFrame(results)