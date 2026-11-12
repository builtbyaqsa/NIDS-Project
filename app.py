import streamlit as st
import pandas as pd
import numpy as np
from src.risk_engine import RiskEngine
from src.xai_explainer import XAIExplainer

# Page configuration
st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("### AI-Powered Threat Detection, Risk Assessment & Batch Analytics")

# Initialize engines
risk_engine = RiskEngine()
explainer = XAIExplainer()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Single Packet Analysis", "Batch CSV Upload Analytics"])

if app_mode == "Single Packet Analysis":
    st.sidebar.header("🔍 Network Packet Input")
    duration = st.sidebar.slider("Duration", 0, 100, 10)
    protocol_type = st.sidebar.selectbox("Protocol Type", [0, 1, 2], format_func=lambda x: ["TCP", "UDP", "ICMP"][x])
    service = st.sidebar.slider("Service Type", 0, 10, 2)
    src_bytes = st.sidebar.number_input("Source Bytes", 0, 10000, 1500)
    dst_bytes = st.sidebar.number_input("Destination Bytes", 0, 10000, 500)

    if st.sidebar.button("Analyze Traffic"):
        prediction = 1 if src_bytes > 3000 else 0
        confidence = 0.92 if prediction == 1 else 0.95
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Threat Assessment")
            risk_result = risk_engine.evaluate_risk(prediction, confidence)
            
            if prediction == 1:
                st.error(f"**Status:** ATTACK DETECTED!")
            else:
                st.success(f"**Status:** NORMAL TRAFFIC")
                
            st.metric(label="Risk Score", value=f"{risk_result['risk_score']} / 100")
            st.write(f"**Risk Level:** {risk_result['risk_level']}")
            st.warning(f"**Recommended Action:** {risk_result['recommended_action']}")
            
        with col2:
            st.subheader("🧠 Explainable AI (XAI) Report")
            feature_names = ['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes']
            input_vals = [duration, protocol_type, service, src_bytes, dst_bytes]
            
            explanation = explainer.explain_prediction(input_vals, feature_names)
            if "top_risk_factors" in explanation:
                st.write("Top contributing factors for this decision:")
                for item in explanation["top_risk_factors"]:
                    st.info(f"**{item['feature']}** — Importance Score: `{item['importance_score']}`")
            else:
                st.write(explanation)
    else:
        st.info("👈 Adjust network parameters in the sidebar and click **Analyze Traffic** to test.")

elif app_mode == "Batch CSV Upload Analytics":
    st.header("📁 Batch Network Traffic Analysis")
    st.markdown("Upload a CSV file containing network connection records to evaluate bulk traffic logs.")
    
    uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("### Preview of Uploaded Data:", df.head())
        
        if st.button("Run Batch Analysis"):
            # Simulating batch prediction logic based on available columns or values
            st.write("---")
            st.subheader("📈 Batch Processing Results")
            
            # Generating dummy risk evaluation for rows
            results = []
            for idx, row in df.iterrows():
                # Simple rule for demo batch
                pred = 1 if 'src_bytes' in df.columns and row['src_bytes'] > 3000 else np.random.choice([0, 1], p=[0.8, 0.2])
                conf = round(np.random.uniform(0.80, 0.99), 2)
                res = risk_engine.evaluate_risk(pred, conf)
                results.append({
                    "Row": idx + 1,
                    "Prediction": "Attack" if pred == 1 else "Normal",
                    "Risk Level": res["risk_level"],
                    "Risk Score": res["risk_score"],
                    "Action": res["recommended_action"]
                })
                
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)
            
            # Summary Metrics
            attacks_count = len(result_df[result_df["Prediction"] == "Attack"])
            normal_count = len(result_df[result_df["Prediction"] == "Normal"])
            
            col1, col2 = st.columns(2)
            col1.metric("Total Normal Packets", normal_count)
            col2.metric("Total Attacks Detected", attacks_count, delta_color="inverse")
    else:
        st.info("Please upload a CSV file to begin batch analysis.")