from prophet import Prophet
from prophet.diagnostics import performance_metrics, cross_validation
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Train a Prophet time series forecasting model
def train_prophet_model(df: pd.DataFrame) -> Prophet:
    # ✅ Ensure required columns are present
    if 'date' not in df.columns or 'quantity' not in df.columns:
        raise ValueError("Dataset must include 'date' and 'quantity' columns.")

    # Aggregate quantity by date and rename columns for Prophet format
    data = df.groupby('date').agg({'quantity': 'sum'}).reset_index()
    data = data.rename(columns={'date': 'ds', 'quantity': 'y'})

    # ✅ Initialize and fit the Prophet model
    model = Prophet()
    model.fit(data)

    return model


# Evaluate the trained Prophet model using historical data
def evaluate_prophet_model(model: Prophet, df: pd.DataFrame, horizon_days=30):
    # Prepare actual data in Prophet format
    data = df.groupby('date').agg({'quantity': 'sum'}).reset_index()
    data = data.rename(columns={'date': 'ds', 'quantity': 'y'})

    # ✅ Create future dataframe and generate forecast
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    # ✅ Merge actuals with forecast to compare
    merged = pd.merge(forecast, data, on='ds', how='inner')

    # ✅ Compute evaluation metrics if ground truth is available
    if not merged.empty:
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = mean_squared_error(merged['y'], merged['yhat'], squared=False)
        mape = mean_absolute_percentage_error(merged['y'], merged['yhat']) * 100

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'Forecast': forecast
        }
    else:
        return {
            'warning': 'Insufficient historical data for evaluation.',
            'Forecast': forecast
        }
