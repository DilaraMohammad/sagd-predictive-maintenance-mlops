import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(
    page_title="SAGD Production Monitoring",
    layout="wide"
)

st.title("SAGD Well Failure MLOps Monitoring Dashboard")
st.markdown("Real-time telemetry and predictive failure monitoring consuming live API endpoints from Kubernetes cluster.")

API_URL = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
auto_refresh = st.sidebar.checkbox("Live Stream Auto-Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Interval (s)", 2, 10, 3)

def fetch_cluster_logs():
    try:
        response = requests.get(f"{API_URL}/history", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []

raw_data = fetch_cluster_logs()

if not raw_data:
    st.warning("Waiting for data stream. Ensure the streamer is running and pushing data to the API service.")
else:
    df = pd.DataFrame(raw_data)

    st.subheader("Latest System Telemetry and Status")
    latest = df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Timestamp", str(latest.get("timestamp", "unknown"))[11:19])
        
    prob = latest.get("failure_probability_20min", 0.0)
    is_alert = prob >= 0.5
        
    with c2:
        risk_label = "Failure Risk" if is_alert else "Normal"
        st.metric("Prediction Flag", risk_label)
        
    with c3:
        status_display = "ALERT" if is_alert else "NORMAL"
        delta_label = "High Risk" if is_alert else "Normal"
        delta_color = "inverse" if is_alert else "normal"
        st.metric("Status", status_display, delta=delta_label, delta_color=delta_color)
        
    with c4:
        st.metric("Failure Probability (20m)", f"{prob:.3f}")

    st.markdown("---")

    st.subheader("Failure Probability Trend")
    if "failure_probability_20min" in df.columns:
        st.line_chart(df[["failure_probability_20min"]])

    st.markdown("---")

    st.subheader("Recent Inference Buffer (Active window)")
    st.dataframe(df, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()