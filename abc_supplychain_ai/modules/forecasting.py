import streamlit as st
import plotly.express as px
from prophet import Prophet
from io import BytesIO
from fpdf import FPDF
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd


# 🔮 Main function to render forecasting interface using Prophet
def render(df):
    st.header("📈 Forecast Demand with Prophet")

    # Normalize column names
    df.columns = [col.lower() for col in df.columns]

    # Validate required columns
    if 'date' not in df.columns or 'quantity' not in df.columns:
        st.error("❌ Data must contain 'date' and 'quantity' columns.")
        return None

    # Aggregate quantity by date and rename columns for Prophet
    data = df.groupby('date').agg({'quantity': 'sum'}).reset_index()
    data = data.rename(columns={"date": "ds", "quantity": "y"})

    # --- User selects forecast range ---
    st.markdown("**Select forecast range:**")
    min_date = data['ds'].min()
    max_date = data['ds'].max()
    default_start = max_date + pd.Timedelta(days=1)
    start_date = st.date_input(
        "Start date for forecast", value=default_start, min_value=min_date, max_value=max_date + pd.Timedelta(days=365))
    horizon = st.number_input(
        "Number of days to forecast (horizon)", min_value=1, max_value=365, value=30)
    custom_range = st.checkbox("Or select custom date range")
    if custom_range:
        range_start = st.date_input(
            "Custom range start date", value=default_start, min_value=min_date, max_value=max_date + pd.Timedelta(days=365), key="range_start")
        range_end = st.date_input(
            "Custom range end date", value=default_start + pd.Timedelta(days=horizon-1),
              min_value=range_start, max_value=max_date + pd.Timedelta(days=365), key="range_end")

    # Train Prophet model
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Evaluate the model on training data
    evaluation = evaluate_model(model, data)
    if evaluation:
        st.subheader("📊 Model Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{evaluation['MAE']:.2f}")
        col2.metric("RMSE", f"{evaluation['RMSE']:.2f}")
        col3.metric("MAPE", f"{evaluation['MAPE']:.2f} %")

    # Line chart for forecasted values
    st.subheader("🔮 Forecast Plot")
    fig1 = px.line(forecast, x='ds', y='yhat', title='Forecasted Demand')
    st.plotly_chart(fig1, use_container_width=True)

    # Confidence interval visualization
    st.subheader("⚖ Confidence Interval")
    fig2 = px.area(forecast, x='ds', y=['yhat_lower', 'yhat_upper'], title='Forecast Confidence Interval')
    st.plotly_chart(fig2, use_container_width=True)

    # --- Filter forecast table by selected range ---
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    if custom_range:
        mask = (forecast['ds'] >= pd.to_datetime(range_start)) & (forecast['ds'] <= pd.to_datetime(range_end))
    else:
        mask = (forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] < pd.to_datetime(start_date) + pd.Timedelta(days=horizon))
    filtered_forecast = forecast.loc[mask, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    st.subheader("📋 Forecast Data Table")
    st.dataframe(filtered_forecast)

    # --- Export options ---
    st.markdown("**Export forecast data:**")
    csv = filtered_forecast.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="forecast_filtered.csv", mime="text/csv")
    excel_bytes = BytesIO()
    filtered_forecast.to_excel(excel_bytes, index=False)
    st.download_button("⬇️ Download Excel", data=excel_bytes.getvalue(), file_name="forecast_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    return forecast


# 📊 Evaluate Prophet model performance on training data
def evaluate_model(model, df: pd.DataFrame):
    try:
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
        y_true = merged['y']
        y_pred = merged['yhat']

        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    except Exception as e:
        st.warning(f"⚠ Model evaluation failed: {e}")
        return None


# 🧾 Generate PDF report from forecast DataFrame
def generate_pdf_table(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Forecast Summary Report", ln=True, align='C')
    pdf.ln(10)

    headers = ["Date", "Forecast", "Lower", "Upper"]
    col_widths = [40, 40, 40, 40]

    # Table header
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, align='C')
    pdf.ln()

    # Table content (limited to first 30 rows)
    for _, row in data.head(30).iterrows():
        pdf.cell(col_widths[0], 10, row['ds'].strftime('%Y-%m-%d'), border=1)
        pdf.cell(col_widths[1], 10, f"{row['yhat']:.2f}", border=1)
        pdf.cell(col_widths[2], 10, f"{row['yhat_lower']:.2f}", border=1)
        pdf.cell(col_widths[3], 10, f"{row['yhat_upper']:.2f}", border=1)
        pdf.ln()

    # Handle output encoding for compatibility
    output = pdf.output(dest='S')
    if isinstance(output, str):
        output = output.encode('latin1')
    return BytesIO(output)


# 📤 Export PDF report button in Streamlit
def render_export_tab(forecast):
    st.header("🧾 Export Forecast Report")
    if forecast is not None:
        pdf_bytes = generate_pdf_table(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        st.download_button("📥 Download Forecast Report (PDF)", data=pdf_bytes, file_name="forecast_report.pdf")
    else:
        st.info("📌 Run forecasting first to enable report export.")
