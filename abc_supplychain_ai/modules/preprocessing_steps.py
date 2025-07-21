# 📁 FILE: modules/preprocessing_steps.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.preprocessing import clean_data

def render(df_raw: pd.DataFrame):
    st.title("🧪 Data Preprocessing Workflow")

    # Step 1: Raw Data Exploration
    st.subheader("📌 Step 1: Raw Data Exploration")
    st.markdown("Inspect the raw uploaded dataset before any preprocessing.")
    st.write(f"Shape of raw data: {df_raw.shape}")
    st.dataframe(df_raw.head())
    st.write("**Data Types:**")
    st.write(df_raw.dtypes)
    st.write("**Missing Values per Column:**")
    st.write(df_raw.isnull().sum())
    st.write("**Number of Duplicate Rows:**")
    st.write(df_raw.duplicated().sum())

    st.markdown("---")

    # Step 2: Data Cleaning
    st.subheader("🧹 Step 2: Data Cleaning")
    df_cleaned = clean_data(df_raw.copy())
    st.success("✔ Data cleaning applied using `clean_data()` function.")
    st.write(f"Shape after cleaning: {df_cleaned.shape}")
    st.dataframe(df_cleaned.head())

    st.markdown("---")

    # Step 3: Data Transformation
    st.subheader("🔁 Step 3: Data Transformation")
    st.info("Column normalization, text standardization, and numeric conversion performed.")
    st.code("""df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    df['sku'] = df['sku'].astype(str).str.upper()
    df['quantity'] = pd.to_numeric(df['quantity'])""")
    st.write("**Updated Data Types:**")
    st.dataframe(df_cleaned.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}))

    st.markdown("---")


    # Step 4: Feature Engineering
    st.subheader("🧠 Step 4: Feature Engineering")
    # Thêm cột tháng từ cột ngày nếu có
    if 'date' in df_cleaned.columns:

        df_cleaned['month'] = pd.to_datetime(df_cleaned['date'], errors='coerce').dt.month
        
        st.write("**Month column extracted from 'date':**")
        st.dataframe(df_cleaned[['date', 'month']].head())

    if 'quantity' in df_cleaned.columns and 'stock' in df_cleaned.columns:
        df_cleaned['sell_through_rate'] = df_cleaned['quantity'] / (df_cleaned['stock'] + 1)
        st.write("**Sell-through Rate:** Quantity / (Stock + 1)")
        st.dataframe(df_cleaned[['sku', 'quantity', 'stock', 'sell_through_rate']].head())
    else:
        st.warning("Columns 'quantity' and 'stock' are required for computing sell-through rate.")

    st.markdown("---")

    # Step 5: Outlier Detection
    st.subheader("🚨 Step 5: Outlier Detection (IQR Method)")
    if 'quantity' in df_cleaned.columns:
        Q1 = df_cleaned['quantity'].quantile(0.25)
        Q3 = df_cleaned['quantity'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df_cleaned[(df_cleaned['quantity'] < lower) | (df_cleaned['quantity'] > upper)]
        st.write(f"Detected {len(outliers)} outlier(s) in 'quantity' based on IQR range.")
        st.dataframe(outliers.head())
    else:
        st.warning("Column 'quantity' is required for outlier analysis.")

    st.markdown("---")

    # Step 6: Feature Scaling
    st.subheader("📏 Step 6: Feature Scaling (Min-Max)")
    scaler = MinMaxScaler()
    scaled_cols = []
    for col in ['quantity', 'stock']:
        if col in df_cleaned.columns:
            df_cleaned[f'{col}_scaled'] = scaler.fit_transform(df_cleaned[[col]])
            scaled_cols.append(f'{col}_scaled')
    if scaled_cols:
        st.write("**Scaled Features Preview:**")
        st.dataframe(df_cleaned[['sku'] + scaled_cols].head())
    else:
        st.info("No numeric features available for scaling.")

    st.markdown("---")

    # Step 7: Data Splitting Preparation
    st.subheader("🧪 Step 7: Data Splitting")
    st.info("Data splitting will be executed within respective modeling modules (RandomForest, Prophet, etc.). No splitting applied here.")

    st.success("🎉 Data preprocessing pipeline completed successfully.")