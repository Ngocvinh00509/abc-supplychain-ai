
import streamlit as st
from modules import overview, forecasting, anomaly, classification, preprocessing_steps
from utils.preprocessing import clean_data, filter_by_sku
import pandas as pd

st.set_page_config(layout="wide", page_title="AI – Demand Forecasting & Alerting")

st.sidebar.title("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sku_filter = st.sidebar.text_input("🔎 Filter by SKU (optional)")
threshold = st.sidebar.slider("🚨 Alert Threshold (Quantity)", 50, 1000, 300)

if not uploaded_file:
    st.warning("📂 Please upload a CSV file to proceed.")
    st.stop()

# Load & tiền xử lý dữ liệu
df_raw = pd.read_csv(uploaded_file)
df = clean_data(df_raw)
df = filter_by_sku(df, sku_filter)
st.success("✅ Data loaded and preprocessed successfully!")

# Tabs for each module
tabs = st.tabs(["🧪 Preprocessing", "📊 Overview", "📈 Forecasting", "📉 Alerts & Anomalies", "🔍 SKU Risk Classification", "🧾 Export Report"])

with tabs[0]:
        preprocessing_steps.render(df_raw)
with tabs[1]:
    overview.render(df)

with tabs[2]:
    forecast = forecasting.render(df)

with tabs[3]:
    anomaly.render(df, threshold)

with tabs[4]:
    classification.render(df)

with tabs[5]:
    forecasting.render_export_tab(forecast)

