import os
import pandas as pd
from src.preprocessing import NIDSDataPreprocessor
from src.utils import setup_logger

logger = setup_logger()

def main():
    logger.info("=== STARTING WEEK 6 DATA PREPROCESSING PIPELINE ===")
    
    RAW_DATA_PATH = "data/raw/nsl_kdd_raw.csv"
    PROCESSED_DIR = "data/processed/"
    
    if not os.path.exists(RAW_DATA_PATH):
        logger.warning(f"Raw dataset not found at '{RAW_DATA_PATH}'. Generating mock NIDS data for validation...")
        os.makedirs("data/raw", exist_ok=True)
        mock_data = pd.DataFrame({
            'duration': [0, 0, 2, 0, 0, 0],
            'protocol_type': ['tcp', 'udp', 'tcp', 'icmp', 'tcp', 'tcp'],
            'src_bytes': [181, 239, 235, 181, 0, 181],
            'dst_bytes': [5450, 486, 1337, 5450, 0, 5450],
            'label': ['normal', 'normal', 'anomaly', 'normal', 'anomaly', 'normal']
        })
        mock_data.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"Sample dataset saved to {RAW_DATA_PATH}")

    categorical_features = ['protocol_type']
    target_attribute = 'label'
    
    preprocessor = NIDSDataPreprocessor(
        categorical_cols=categorical_features, 
        target_col=target_attribute
    )

    raw_df = preprocessor.load_data(RAW_DATA_PATH)
    cleaned_df = preprocessor.clean_duplicates_and_nulls(raw_df)
    X_processed, y_processed = preprocessor.encode_and_scale(cleaned_df, is_training=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    processed_final = pd.concat([X_processed, y_processed], axis=1)
    processed_final.to_csv(os.path.join(PROCESSED_DIR, "cleaned_nids_data.csv"), index=False)
    
    preprocessor.save_artifacts(PROCESSED_DIR)

    logger.info(f"Processed dataset saved to '{PROCESSED_DIR}cleaned_nids_data.csv'.")
    logger.info(f"Final Data Matrix Shape: {X_processed.shape}")
    logger.info("=== WEEK 6 PREPROCESSING PIPELINE EXECUTED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()