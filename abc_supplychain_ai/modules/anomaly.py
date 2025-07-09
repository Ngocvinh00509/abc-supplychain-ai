import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest
import pandas as pd

# 🚨 Main function to render alerts and anomaly detection tab
def render(df: pd.DataFrame, threshold: float):
    st.header("🚨 Alerts & 📊 Anomaly Detection")

    # 🧩 Step 1: Ensure 'quantity' column exists
    if 'quantity' not in df.columns:
        st.error("❌ 'quantity' column is missing.")
        return

    # ✅ Step 2: Handle missing values in 'quantity'
    if df['quantity'].isnull().any():
        st.warning("⚠ Missing values found in 'quantity'. Filling with 0.")
        df['quantity'] = df['quantity'].fillna(0)

    # 📌 Show quantity range
    st.caption(f"📌 Quantity range: {df['quantity'].min():.2f} → {df['quantity'].max():.2f}")

    # 🚨 Step 3: Threshold-based alerts
    alerts = df[df['quantity'] > threshold]
    st.subheader("🚨 Quantity Threshold Alerts")
    st.info(f"{len(alerts)} records exceeded the threshold of {threshold}")
    st.dataframe(alerts)

    # 🧠 Step 4: Anomaly detection using Isolation Forest
    try:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly_score'] = iso.fit_predict(df[['quantity']])
        df['anomaly'] = df['anomaly_score'] == -1  # Label anomalies as True
    except Exception as e:
        st.error(f"❌ IsolationForest failed: {e}")
        return

    # 📈 Step 4b: Model Evaluation Summary
    st.subheader("📈 Model Evaluation Summary")
    anomaly_count = df['anomaly'].sum()
    total_records = len(df)
    contamination_ratio = anomaly_count / total_records

    st.markdown(f"""
    - **Anomalies Detected**: `{anomaly_count}` / `{total_records}` records  
    - **Detected Contamination Ratio**: `{contamination_ratio:.2%}`  
    - **Model Contamination Parameter**: `{iso.contamination:.2%}`
    """)

    # 📋 Step 5: Display detected anomalies
    anomalies = df[df['anomaly']]
    st.subheader("🧭 Anomaly Records")
    if anomalies.empty:
        st.info("✅ No anomalies detected.")
    else:
        st.dataframe(anomalies)

    # 📈 Step 6: Time series plot of quantity with anomalies
    if 'date' in df.columns:
        st.subheader("📉 Quantity Over Time (with Anomalies)")
        fig_time = px.line(
            df, x='date', y='quantity', color='anomaly',
            title="Quantity Trend with Anomaly Highlight",
            color_discrete_map={False: 'blue', True: 'red'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("ℹ 'date' column is missing. Time series plot not available.")

    # 📊 Step 7: Histogram of quantity distribution
    st.subheader("📊 Quantity Distribution")
    fig_hist = px.histogram(
        df, x='quantity', nbins=30, title="Histogram of Quantity"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
