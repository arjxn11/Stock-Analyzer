import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np

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
        df = df.reset_index()
        st.subheader("✅ Raw Data:")
        st.dataframe(df.head())

        # Test VWAP only
        if 'Close' in df.columns and 'Volume' in df.columns:
            volume_cumsum = df['Volume'].replace(0, np.nan).cumsum()
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / volume_cumsum
            st.subheader("📈 VWAP Sample:")
            st.dataframe(df[['Date', 'Close', 'Volume', 'VWAP']].tail(10))
        else:
            st.warning("Volume or Close column missing — cannot compute VWAP.")