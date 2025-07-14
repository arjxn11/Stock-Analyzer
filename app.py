import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer (with VWAP & TWAP)")

# --- User Inputs ---
ticker = st.text_input("Enter stock ticker (e.g., AAPL, NVDA):").upper().strip()
start = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end = st.date_input("End Date", value=datetime.today())

# --- Fetch Data ---
@st.cache_data
def get_stock_data(tkr, st_dt, en_dt):
    if not tkr:
        return pd.DataFrame()
    df = yf.download(tkr, start=st_dt, end=en_dt, interval="1d")  # <— NO group_by
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

# --- VWAP Calculation ---
def calculate_vwap(df):
    if 'Close' in df.columns and 'Volume' in df.columns:
        volume = df['Volume'].replace(0, np.nan)
        df['VWAP'] = (df['Close'] * volume).cumsum() / volume.cumsum()
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

# --- Main Logic ---
if st.button("Analyze"):
    df = get_stock_data(ticker, start, end)

    if df.empty:
        st.error("❌ No data returned. Check the ticker symbol or date range.")
    else:
        # Clean column names just in case
        df.columns = df.columns.astype(str).str.strip()

        # Show raw data
        st.subheader("✅ Raw Extracted Data")
        st.dataframe(df.head())

        # Add VWAP & TWAP
        df = calculate_vwap(df)
        df = calculate_twap(df)

        # Preview metrics
        st.subheader("📈 VWAP & TWAP Preview")
        st.dataframe(df[['Date', 'Close', 'Volume', 'VWAP', 'TWAP']].tail())

        # Plot
        st.subheader("📉 Price Chart")
        try:
            df_plot = df.set_index("Date")[['Close', 'VWAP', 'TWAP']]
            st.line_chart(df_plot)
        except KeyError:
            st.warning("Cannot plot — one or more of Close/VWAP/TWAP not found.")
