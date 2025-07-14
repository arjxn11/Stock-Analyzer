import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

# Streamlit page setup
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📄 Raw Stock Data Extractor")

# --- User Inputs ---
ticker = st.text_input("Enter stock ticker (e.g., AAPL, NVDA):").upper().strip()
start = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end = st.date_input("End Date", value=datetime.today())

# --- Fetch Data ---
@st.cache_data
def get_stock_data(ticker, start, end):
    if ticker == "":
        return pd.DataFrame()
    df = yf.download(ticker, start=start, end=end, interval="1d")
    df.dropna(inplace=True)
    return df

# --- Main Execution ---
if st.button("Analyze"):
    df = get_stock_data(ticker, start, end)

    if df.empty:
        st.error("❌ No data returned. Check the ticker symbol or date range.")
    else:
        df = df.reset_index()  # Move 'Date' from index to column
        st.subheader("✅ Extracted Stock Data (Index Reset)")
        st.dataframe(df.head(50))  # Show first 50 rows
