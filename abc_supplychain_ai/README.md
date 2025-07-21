# AI Supply Chain Analytics Website

## Overview

This project is a web application built with Streamlit for analyzing, forecasting demand, and detecting anomalies in the supply chain based on sales/SKU data. The app integrates Machine Learning models such as Prophet (forecasting), Isolation Forest (anomaly detection), Random Forest (SKU risk classification), and data visualization with Plotly.

## Main Features

- **CSV Data Upload**: Supports filtering by SKU and setting alert thresholds.
- **Data Overview**: Displays data tables and charts by date, SKU, region, and event.
- **Demand Forecasting**: Uses Prophet to forecast future sales quantity, with visual charts and forecast tables.
- **Alerts & Anomaly Detection**: Automatically detects values exceeding thresholds and anomalies using Isolation Forest.
- **SKU Risk Classification**: Classifies SKUs into risk groups (Overstock, Balanced, Understock) using Random Forest, with pie/bar charts and model evaluation metrics.
- **Export PDF Report**: Download forecast reports as PDF files.

## Folder Structure

```
abc_supplychain_ai/
    app.py                # Main Streamlit application file
    config.py             # (Optional) General configuration
    requirements.txt      # List of required Python libraries
    modules/
        classification.py # SKU risk classification module
models/                  # (Optional) Store ML models or sample data
```

## Installation Guide

1. **Clone the repository**

```bash
git clone <repo-url>
cd analyticswebsite/abc_supplychain_ai
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
```

3. **Install required libraries**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
streamlit run app.py
```

## Input Data Requirements

- The CSV file must contain at least the following columns: `Date`, `Quantity`, `SKU`, `Stock`.
- Additional columns such as `Region`, `Event` will enable deeper analysis.

## Technologies Used

- Python, Streamlit, Pandas, Plotly, Prophet, scikit-learn, FPDF

## Contribution

- Pull requests and issues are always welcome!
- Please clearly describe the problem or feature you want to contribute.

## Author

- DO NGOC VINH

---

> This application helps businesses proactively forecast demand, detect risks, and optimize supply chain inventory using AI.
