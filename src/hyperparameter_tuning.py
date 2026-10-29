import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import setup_logger

logger = setup_logger()

class NIDSModelTuner:
    """
    Performs hyperparameter tuning for the optimal NIDS model using GridSearchCV
    to maximize detection accuracy and robustness.
    """
    def __init__(self, data_path: str, features_path: str):
        self.data_path = data_path
        self.features_path = features_path

    def load_data(self):
        """Loads processed data, applies selected features, and splits into train/test."""
        logger.info("Loading data for hyperparameter tuning...")
        df = pd.read_csv(self.data_path)
        
        if os.path.exists(self.features_path):
            selected_features = joblib.load(self.features_path)
            if 'label' not in selected_features:
                selected_features.append('label')
            df = df[selected_features]
            logger.info(f"Loaded {len(selected_features)-1} features for tuning.")
            
        X = df.drop(columns=['label'])
        y = df['label']
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def tune_random_forest(self, X_train, y_train):
        """Sets up and runs Grid Search to find the best hyperparameters for Random Forest."""
        logger.info("Configuring GridSearchCV parameters for Random Forest...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        logger.info("Running Grid Search optimization (training multiple variations)...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best Hyperparameters Found: {grid_search.best_params_}")
        logger.info(f"Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")
        
        return grid_search.best_estimator_

    def save_tuned_model(self, model, output_dir: str = "data/processed/"):
        """Saves the optimized best model artifact."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "tuned_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Tuned model successfully saved to {model_path}")