import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np

# Streamlit setup
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer (with VWAP & TWAP)")

# --- Inputs ---
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

# --- VWAP Calculation ---
def calculate_vwap(df):
    if 'Close' in df.columns and 'Volume' in df.columns:
        volume_cumsum = df['Volume'].replace(0, np.nan).cumsum()
        if volume_cumsum.isnull().all():
            df['VWAP'] = np.nan
        else:
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / volume_cumsum
    else:
        df['VWAP'] = np.nan
    return df

# --- TWAP Calculation ---
def calculate_twap(df):
    if 'Close' in df.columns:
        df['TWAP'] = df['Close'].expanding().mean()
    else:
        df['TWAP'] = np.nan
    return df

# --- Main Execution ---
if st.button("Analyze"):
    df = get_stock_data(ticker, start, end)

    if df.empty:
        st.error("❌ No data returned. Check the ticker symbol or date range.")
    else:
        df = df.reset_index()

        st.subheader("✅ Raw Extracted Data")
        st.dataframe(df.head())

        # --- VWAP & TWAP ---
        df = calculate_vwap(df)
        df = calculate_twap(df)

        st.subheader("📈 VWAP & TWAP Preview")
        st.dataframe(df[['Date', 'Close', 'Volume', 'VWAP', 'TWAP']].tail(10))

        # Optional: Add line chart
        plot_cols = ['Close', 'VWAP', 'TWAP']
        if all(col in df.columns and df[col].notna().any() for col in plot_cols):
            st.subheader("📉 Price Chart")
            st.line_chart(df[plot_cols])
        else:
            st.warning("Cannot plot — missing or invalid data in Close/VWAP/TWAP.")
