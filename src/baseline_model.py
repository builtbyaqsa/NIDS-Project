import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import setup_logger

logger = setup_logger()

class NIDSBaselineModel:
    """
    A baseline machine learning model for Network Intrusion Detection.
    Trains a Random Forest classifier using Week 7 selected features to establish 
    a performance benchmark.
    """
    def __init__(self, data_path: str, features_path: str):
        self.data_path = data_path
        self.features_path = features_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data_and_features(self):
        """Loads cleaned data and filters out only the top features selected in Week 7."""
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset not found at {self.data_path}")
            raise FileNotFoundError(f"File not found: {self.data_path}")
            
        logger.info(f"Loading processed dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        if os.path.exists(self.features_path):
            selected_features = joblib.load(self.features_path)
            logger.info(f"Loaded {len(selected_features)} selected features from Week 7.")
            # Ensure target column 'label' is kept
            if 'label' not in selected_features:
                selected_features.append('label')
            df = df[selected_features]
        else:
            logger.warning("Selected features file not found. Utilizing all available features.")

        # Separate features (X) and target (y)
        X = df.drop(columns=['label'])
        y = df['label']

        # Train-Test Split (80% training, 20% testing with stratification)
        logger.info("Splitting dataset into training and testing sets (80/20)...")
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Trains the Random Forest model and calculates evaluation metrics."""
        logger.info("Training baseline Random Forest model...")
        self.model.fit(X_train, y_train)

        logger.info("Evaluating model performance on test set...")
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        logger.info(f"Baseline Model Accuracy Achieved: {acc * 100:.2f}%")
        return acc, report, conf_matrix

    def save_model(self, output_dir: str = "data/processed/"):
        """Saves the trained baseline model artifact for future comparison."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "baseline_model.pkl")
        joblib.dump(self.model, model_path)
        logger.info(f"Baseline model successfully saved to {model_path}")