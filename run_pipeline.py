import os
import pandas as pd
from src.risk_engine import RiskEngine
from src.xai_explainer import XAIExplainer

def run_nids_pipeline():
    print("="*50)
    print("🚀 STARTING UNIFIED NIDS PIPELINE")
    print("="*50)
    
    # 1. Initialize Risk Engine
    risk_engine = RiskEngine()
    
    # 2. Initialize XAI Explainer
    explainer = XAIExplainer()
    
    # 3. Simulate processing incoming traffic sample
    # (In later weeks, this will read from live network streams or test batches)
    print("\n[INFO] Analyzing incoming network traffic sample...")
    
    # Example prediction: 1 (Attack detected), Confidence: 0.92
    sample_prediction = 1
    sample_confidence = 0.92
    
    # Evaluate Risk
    risk_result = risk_engine.evaluate_risk(sample_prediction, sample_confidence)
    print("\n--- RISK ASSESSMENT RESULT ---")
    print(f"Risk Level        : {risk_result['risk_level']}")
    print(f"Risk Score        : {risk_result['risk_score']}")
    print(f"Recommended Action: {risk_result['recommended_action']}")
    
    # XAI Explanation
    dummy_features = ['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes']
    explanation = explainer.explain_prediction([1, 0, 2, 450, 120], dummy_features)
    print("\n--- XAI EXPLAINABILITY REPORT ---")
    print(f"Top Risk Factors  : {explanation}")
    
    print("\n" + "="*50)
    print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    run_nids_pipeline()