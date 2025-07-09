import streamlit as st
import plotly.express as px

def render(df):
    st.header("📊 Data Overview")
    st.dataframe(df.head())

    # Normalize all column names to lowercase (as a precaution)
    df.columns = [col.lower() for col in df.columns]

    # Aggregate by Region if available
    if 'region' in df.columns and 'quantity' in df.columns:
        fig_region = px.bar(
            df.groupby('region')['quantity'].sum().reset_index(),
            x='region', y='quantity',
            title='📍 Quantity by Region'
        )
        st.plotly_chart(fig_region, use_container_width=True)

    # Aggregate by SKU if available
    if 'sku' in df.columns and 'quantity' in df.columns:
        fig_sku = px.bar(
            df.groupby('sku')['quantity'].sum().reset_index(),
            x='sku', y='quantity',
            title='📦 Quantity by SKU'
        )
        st.plotly_chart(fig_sku, use_container_width=True)

    # Plot trend over time
    if 'date' in df.columns and 'quantity' in df.columns:
        fig_daily = px.line(
            df.groupby('date')['quantity'].sum().reset_index(),
            x='date', y='quantity',
            title='📈 Daily Order Trend'
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    # Analyze quantity distribution by event type if available
    if 'event' in df.columns and 'quantity' in df.columns:
        fig_event = px.box(
            df, x='event', y='quantity',
            title='🧊 Quantity vs Event Type'
        )
        st.plotly_chart(fig_event, use_container_width=True)
