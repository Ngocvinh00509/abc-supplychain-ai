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

    # Train Prophet model
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
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

    # Display forecast data
    st.subheader("📋 Forecast Data Table")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

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
