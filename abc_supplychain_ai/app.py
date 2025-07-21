
import streamlit as st
from modules import overview, forecasting, anomaly, classification, preprocessing_steps
from utils.preprocessing import clean_data, filter_by_sku
import pandas as pd

st.set_page_config(layout="wide", page_title="AI – Demand Forecasting & Alerting")

st.sidebar.title("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sku_filter = st.sidebar.text_input("🔎 Filter by SKU (optional)")
threshold = st.sidebar.slider("🚨 Alert Threshold (Quantity)", 50, 1000, 300)


import os
default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'abc_sales_orders_2024.csv')
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded from uploaded file!")
elif os.path.exists(default_path):
    df_raw = pd.read_csv(default_path)
    st.info(f"ℹ Using default data file: {default_path}")
else:
    st.warning("📂 Please upload a CSV file to proceed.")
    st.stop()

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

