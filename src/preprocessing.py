import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path: str, output_path: str):
    """
    Loads raw NIDS data, performs cleaning, label encoding, 
    scaling, and saves the processed dataset.
    """
    print("[INFO] Loading raw dataset...")
    if not os.path.exists(input_path):
        print(f"[ERROR] Raw dataset not found at: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    
    # Basic cleaning
    print("[INFO] Cleaning data (handling missing values)...")
    df.fillna(method='ffill', inplace=True)
    
    # Encoding categorical columns (example fields common in NIDS datasets)
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    # Saving processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Preprocessed data saved successfully to {output_path}")

if __name__ == "__main__":
    # Test path configuration
    raw_file = "data/raw/nids_raw_data.csv"
    processed_file = "data/processed/cleaned_nids_data.csv"
    preprocess_data(raw_file, processed_file)