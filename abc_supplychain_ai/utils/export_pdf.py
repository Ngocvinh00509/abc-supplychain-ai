# 📁 FILE: utils/export_pdf.py

from fpdf import FPDF
from io import BytesIO
import pandas as pd

def generate_forecast_pdf(data: pd.DataFrame) -> BytesIO:
    """
    Generate a PDF report containing the forecast summary.

    Parameters:
        data (pd.DataFrame): Forecast DataFrame containing columns 'ds', 'yhat', 'yhat_lower', 'yhat_upper'

    Returns:
        BytesIO: In-memory PDF binary stream ready for download or sending via API
    """
    # Initialize PDF document
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(0, 10, "Forecast Summary Report", ln=True, align='C')
    pdf.ln(10)

    # Define column widths and headers
    col_widths = [40, 40, 40, 40]
    headers = ["Date", "Forecast", "Lower", "Upper"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, align='C')
    pdf.ln()

    # Write forecast rows (limit to first 30 entries)
    for _, row in data.head(30).iterrows():
        pdf.cell(col_widths[0], 10, row['ds'].strftime('%Y-%m-%d'), border=1)
        pdf.cell(col_widths[1], 10, f"{row['yhat']:.2f}", border=1)
        pdf.cell(col_widths[2], 10, f"{row['yhat_lower']:.2f}", border=1)
        pdf.cell(col_widths[3], 10, f"{row['yhat_upper']:.2f}", border=1)
        pdf.ln()

    # Export to byte stream (compatible with Streamlit download button)
    pdf_output = pdf.output(dest='S').encode('latin1')  # Encode as byte string
    return BytesIO(pdf_output)  # Return as in-memory file-like object
