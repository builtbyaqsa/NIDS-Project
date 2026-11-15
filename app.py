import time
import streamlit as st
import pandas as pd
import numpy as np
from src.risk_engine import RiskEngine
from src.xai_explainer import XAIExplainer
from src.database import init_db, log_prediction, fetch_logs, clear_logs
from src.model_loader import load_trained_model, predict_traffic


def build_incident_report(logs_df: pd.DataFrame) -> str:
    total = len(logs_df)
    attacks = int((logs_df["prediction"] == "Attack").sum()) if total else 0
    normals = int((logs_df["prediction"] == "Normal").sum()) if total else 0
    avg_risk = round(float(logs_df["risk_score"].mean()), 2) if total else 0.0

    lines = [
        "# NIDS Security Incident & Telemetry Report",
        "",
        "## Summary",
        f"- Total Events: {total}",
        f"- Attacks Detected: {attacks}",
        f"- Normal Traffic: {normals}",
        f"- Average Risk Score: {avg_risk} / 100",
        "",
        "## Recent Attacks",
    ]

    attacks_df = logs_df[logs_df["prediction"] == "Attack"].head(10)
    if attacks_df.empty:
        lines.append("- No attacks recorded.")
    else:
        for _, row in attacks_df.iterrows():
            lines.append(
                f"- [{row['timestamp']}] {row['mode']} | Score: {row['risk_score']} | Level: {row['risk_level']} | Action: {row['recommended_action']}"
            )

    return "\n".join(lines)


init_db()

st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("### AI-Powered Threat Detection, Risk Assessment, XAI & SOC Telemetry")

risk_engine = RiskEngine()
explainer = XAIExplainer()
model = load_trained_model()

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "Single Packet Analysis",
        "Batch CSV Upload Analytics",
        "Live Threat Simulation",
        "Telemetry & Audit Logs",
    ],
)

if model is not None:
    st.sidebar.success("Model Loaded: tuned_model.pkl")
    model_type = type(model).__name__
    n_features = getattr(model, "n_features_in_", "N/A")
    st.sidebar.caption(f"Estimator: `{model_type}` | Features: `{n_features}`")
else:
    st.sidebar.warning("Model file not found — using fallback mode")

if app_mode == "Single Packet Analysis":
    st.sidebar.header("🔍 Network Packet Input")
    duration = st.sidebar.slider("Duration", 0, 100, 10)
    protocol_type = st.sidebar.selectbox(
        "Protocol Type", [0, 1, 2], format_func=lambda x: ["TCP", "UDP", "ICMP"][x]
    )
    service = st.sidebar.slider("Service Type", 0, 10, 2)
    src_bytes = st.sidebar.number_input("Source Bytes", 0, 10000, 1500)
    dst_bytes = st.sidebar.number_input("Destination Bytes", 0, 10000, 500)

    if st.sidebar.button("Analyze Traffic"):
        feature_names = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes"]
        input_vals = [duration, protocol_type, service, src_bytes, dst_bytes]
        feature_array = np.array([input_vals], dtype=float)

        prediction_val, confidence, engine_used = predict_traffic(model, feature_array)
        risk_result = risk_engine.evaluate_risk(prediction_val, confidence)
        pred_label = "Attack" if prediction_val == 1 else "Normal"

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Threat Assessment")
            if prediction_val == 1:
                st.error("**Status:** ATTACK DETECTED!")
            else:
                st.success("**Status:** NORMAL TRAFFIC")

            st.caption(f"Inference Engine: `{engine_used}` | Confidence: `{confidence}`")
            st.metric(label="Risk Score", value=f"{risk_result['risk_score']} / 100")
            st.write(f"**Risk Level:** {risk_result['risk_level']}")
            st.warning(f"**Recommended Action:** {risk_result['recommended_action']}")

            log_prediction(
                "Single Packet",
                pred_label,
                risk_result["risk_score"],
                risk_result["risk_level"],
                risk_result["recommended_action"],
            )

        with col2:
            st.subheader("🧠 Explainable AI (XAI) Report")
            explanation = explainer.explain_prediction(input_vals, feature_names)
            if isinstance(explanation, dict) and "top_risk_factors" in explanation:
                st.write("Top contributing factors for this decision:")
                for item in explanation["top_risk_factors"]:
                    st.info(
                        f"**{item['feature']}** — Importance Score: `{item['importance_score']}`"
                    )
            else:
                st.write(explanation)
    else:
        st.info("👈 Adjust network parameters in the sidebar and click **Analyze Traffic** to test.")

elif app_mode == "Batch CSV Upload Analytics":
    st.header("📁 Batch Network Traffic Analysis")
    st.markdown(
        "Upload a CSV file containing network connection records to evaluate bulk traffic logs."
    )

    uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded CSV is empty.")
        else:
            st.success("File uploaded successfully!")
            st.write("### Preview of Uploaded Data:", df.head())

            if st.button("Run Batch Analysis"):
                st.subheader("📈 Batch Processing Results")
                results = []
                required_cols = [
                    "duration",
                    "protocol_type",
                    "service",
                    "src_bytes",
                    "dst_bytes",
                ]
                has_required = all(col in df.columns for col in required_cols)

                for idx, row in df.iterrows():
                    if has_required:
                        row_arr = np.array([[row[c] for c in required_cols]], dtype=float)
                        pred, conf, _ = predict_traffic(model, row_arr)
                    else:
                        src_val = row["src_bytes"] if "src_bytes" in df.columns else 0
                        pred = 1 if src_val > 3000 else 0
                        conf = 0.92 if pred == 1 else 0.95

                    res = risk_engine.evaluate_risk(pred, conf)
                    pred_label = "Attack" if pred == 1 else "Normal"

                    results.append(
                        {
                            "Row": idx + 1,
                            "Prediction": pred_label,
                            "Confidence": conf,
                            "Risk Level": res["risk_level"],
                            "Risk Score": res["risk_score"],
                            "Action": res["recommended_action"],
                        }
                    )
                    log_prediction(
                        "Batch CSV",
                        pred_label,
                        res["risk_score"],
                        res["risk_level"],
                        res["recommended_action"],
                    )

                result_df = pd.DataFrame(results)
                st.dataframe(result_df, use_container_width=True)

                attacks_count = int((result_df["Prediction"] == "Attack").sum())
                normal_count = int((result_df["Prediction"] == "Normal").sum())

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Packets", len(result_df))
                c2.metric("Normal Traffic", normal_count)
                c3.metric("Attacks Detected", attacks_count)

                st.bar_chart(result_df["Prediction"].value_counts())

                csv_export = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Batch Results CSV",
                    data=csv_export,
                    file_name="nids_batch_results.csv",
                    mime="text/csv",
                )
    else:
        st.info("Please upload a CSV file to begin batch analysis.")

elif app_mode == "Live Threat Simulation":
    st.header("📡 Live SOC Threat Simulation Stream")
    st.markdown(
        "Simulate incoming network packets and automatically log high-risk alerts to SQLite."
    )

    col_a, col_b = st.columns(2)
    packet_count = col_a.slider("Packets to Simulate", 5, 30, 10)
    delay_sec = col_b.slider("Interval (seconds)", 0.1, 1.0, 0.3)

    if st.button("Start Live Stream"):
        stream_placeholder = st.empty()
        live_rows = []

        for i in range(1, packet_count + 1):
            sample = np.array(
                [
                    [
                        np.random.randint(0, 80),
                        np.random.choice([0, 1, 2]),
                        np.random.randint(0, 10),
                        np.random.choice([400, 1200, 4500, 7800]),
                        np.random.choice([200, 900, 3500]),
                    ]
                ],
                dtype=float,
            )
            pred, conf, _ = predict_traffic(model, sample)
            res = risk_engine.evaluate_risk(pred, conf)
            pred_label = "Attack" if pred == 1 else "Normal"

            log_prediction(
                "Live Simulation",
                pred_label,
                res["risk_score"],
                res["risk_level"],
                res["recommended_action"],
            )

            live_rows.append(
                {
                    "Packet #": i,
                    "src_bytes": int(sample[0][3]),
                    "Prediction": pred_label,
                    "Risk Level": res["risk_level"],
                    "Risk Score": res["risk_score"],
                }
            )
            stream_placeholder.dataframe(pd.DataFrame(live_rows), use_container_width=True)
            time.sleep(delay_sec)

        st.success(f"Completed {packet_count} simulated packets and logged to SQLite.")

elif app_mode == "Telemetry & Audit Logs":
    st.header("📋 System Telemetry & Audit Logs")
    st.markdown("Historical record of all analyzed traffic and detected threats stored in SQLite.")

    logs_df = fetch_logs()
    if not logs_df.empty:
        filter_pred = st.selectbox("Filter by Prediction", ["All", "Attack", "Normal"])
        if filter_pred != "All":
            logs_df = logs_df[logs_df["prediction"] == filter_pred]

        m1, m2, m3 = st.columns(3)
        m1.metric("Displayed Events", len(logs_df))
        m2.metric("Attacks in View", int((logs_df["prediction"] == "Attack").sum()))
        m3.metric("Avg Risk Score", round(float(logs_df["risk_score"].mean()), 2))

        st.subheader("Threat Distribution")
        st.bar_chart(logs_df["prediction"].value_counts())

        st.subheader("Audit Log Table")
        st.dataframe(logs_df, use_container_width=True)

        c_btn1, c_btn2, c_btn3 = st.columns(3)
        csv_logs = logs_df.to_csv(index=False).encode("utf-8")
        c_btn1.download_button(
            label="⬇️ Export Audit Logs (CSV)",
            data=csv_logs,
            file_name="nids_audit_logs.csv",
            mime="text/csv",
        )
        if c_btn2.button("Refresh Logs"):
            st.rerun()
        if c_btn3.button("🗑️ Clear All Logs"):
            clear_logs()
            st.rerun()

        report_text = build_incident_report(logs_df)
        st.download_button(
            label="📄 Download Incident Report (.md)",
            data=report_text.encode("utf-8"),
            file_name="nids_incident_report.md",
            mime="text/markdown",
        )
    else:
        st.info("No logs recorded yet. Run some packet analyses, batch tests, or live simulation!")