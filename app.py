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
def get_stock_data(ticker, start, end):
    if ticker == "":
        return pd.DataFrame()
    df = yf.download(ticker, start=start, end=end, interval="1d", group_by="ticker")
    df.dropna(inplace=True)
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

# --- Main Execution ---
if st.button("Analyze"):
    df = get_stock_data(ticker, start, end)

    if df.empty:
        st.error("❌ No data returned. Check the ticker symbol or date range.")
    else:
        df = df.reset_index()

        # --- Handle MultiIndex (e.g., from group_by="ticker") ---
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(1):
                df = df.xs(ticker, axis=1, level=1)
                df.columns = df.columns.str.strip()
        elif isinstance(df.columns, pd.Index):
            df.columns = df.columns.str.strip()

        st.subheader("✅ Raw Extracted Data")
        st.dataframe(df.head())

        # --- Calculate VWAP & TWAP ---
        df = calculate_vwap(df)
        df = calculate_twap(df)

        st.subheader("📈 VWAP & TWAP Preview")
        st.dataframe(df[['Date', 'Close', 'Volume', 'VWAP', 'TWAP']].tail())

        # --- Plot Safely ---
        def is_column_valid(df, col):
            return col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any()

        plot_cols = ['Close', 'VWAP', 'TWAP']
        valid_plot_cols = [col for col in plot_cols if is_column_valid(df, col)]

        if valid_plot_cols:
            st.subheader("📉 Price Chart")
            st.line_chart(df[valid_plot_cols])
        else:
            st.warning("Cannot plot — one or more of Close/VWAP/TWAP are missing or contain only NaNs.")
