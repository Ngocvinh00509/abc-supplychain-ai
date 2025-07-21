import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 🔹 Normalize column names: lowercase, strip whitespace, convert to snake_case
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    # 🔹 Convert 'date' column to datetime (if exists), remove invalid entries
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    # 🔹 Standardize text format for SKU, Region, Event
    for col in ['sku', 'region', 'event']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    # 🔹 Convert quantity and stock columns to numeric values
    for col in ['quantity', 'stock']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 🔹 Drop rows with missing values in required business columns
    required_cols = ['sku', 'quantity', 'stock']
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    
    # 🔹 Remove invalid (negative) quantity or stock values
    if 'quantity' in df.columns and 'stock' in df.columns:
        df = df[(df['quantity'] >= 0) & (df['stock'] >= 0)]
    # 🔹 Remove completely duplicated rows
    df = df.drop_duplicates()

    return df


def filter_by_sku(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    """
    Filter the dataset for a specific SKU (after text normalization).

    Parameters:
        df (pd.DataFrame): Cleaned dataset
        sku (str): SKU code to filter

    Returns:
        pd.DataFrame: Subset of data matching the SKU
    """
    if 'sku' in df.columns and sku:
        return df[df['sku'] == sku.upper().strip()]
    return df
