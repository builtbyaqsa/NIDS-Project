import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    print("[INFO] Generating sample training data for NIDS...")
    
    # Creating a synthetic dataset for pipeline testing (simulating KDD / NSL-KDD features)
    np.random.seed(42)
    n_samples = 1000
    
    X_data = {
        'duration': np.random.randint(0, 100, n_samples),
        'protocol_type': np.random.randint(0, 3, n_samples),
        'service': np.random.randint(0, 10, n_samples),
        'src_bytes': np.random.randint(100, 5000, n_samples),
        'dst_bytes': np.random.randint(100, 5000, n_samples)
    }
    
    df = pd.DataFrame(X_data)
    # Target label: 0 for normal, 1 for attack
    df['Label'] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    X = df[['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes']]
    y = df['Label']
    
    print("[INFO] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Ensure processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    # Save the trained model
    model_path = "data/processed/tuned_model.pkl"
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Model trained and saved successfully at: {model_path}")

if __name__ == "__main__":
    train_and_save_model()