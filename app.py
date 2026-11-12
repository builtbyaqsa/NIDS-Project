import streamlit as st
import pandas as pd
from src.risk_engine import RiskEngine
from src.xai_explainer import XAIExplainer

# Page configuration
st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("### AI-Powered Threat Detection, Risk Assessment & XAI Explainer")

# Initialize engines
risk_engine = RiskEngine()
explainer = XAIExplainer()

# Sidebar inputs for testing traffic
st.sidebar.header("🔍 Network Packet Input")
duration = st.sidebar.slider("Duration", 0, 100, 10)
protocol_type = st.sidebar.selectbox("Protocol Type", [0, 1, 2], format_func=lambda x: ["TCP", "UDP", "ICMP"][x])
service = st.sidebar.slider("Service Type", 0, 10, 2)
src_bytes = st.sidebar.number_input("Source Bytes", 0, 10000, 1500)
dst_bytes = st.sidebar.number_input("Destination Bytes", 0, 10000, 500)

if st.sidebar.button("Analyze Traffic"):
    # Simulated prediction based on inputs (or model prediction)
    # For demo, if src_bytes > 3000, treat as attack (1), else normal (0)
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