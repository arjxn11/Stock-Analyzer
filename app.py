import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer (with VWAP & TWAP)")

# Request and receive input
ticker = st.text_input("Enter stock ticker (e.g., AAPL, NVDA):").upper().strip()
start = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end = st.date_input("End Date", value=datetime.today())


# Data extraction
@st.cache_data
def get_stock_data(ticker, start, end):
    if ticker == "":
        return pd.DataFrame()
    df = yf.download(ticker, start=start, end=end, interval="1d")
    df.dropna(inplace=True)
    return df


# Calculating VWAP and TWAP
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


def calculate_twap(df):
    if 'Close' in df.columns:
        df['TWAP'] = df['Close'].expanding().mean()
    else:
        df['TWAP'] = np.nan
    return df


# Analysis
if st.button("Analyze"):

    df = get_stock_data(ticker, start, end)

    if df.empty:
        st.error("❌ No data returned. Check the ticker symbol or date range.")
    else:
        df.columns = df.columns.str.strip()  # Normalize column names
        df = calculate_vwap(df)
        df = calculate_twap(df)

        wanted_cols = ['Close', 'VWAP', 'TWAP']

        # Check both column presence and if they're not all NaNs
        if all(col in df.columns for col in wanted_cols) and all(df[col].notna().sum() > 0 for col in wanted_cols):
            st.line_chart(df[wanted_cols])
            st.subheader("📋 Latest Data")
            st.dataframe(df[['Close', 'Volume', 'VWAP', 'TWAP']].tail())
        else:
            st.warning("⚠️ One or more of Close/VWAP/TWAP are missing or contain only NaNs.")
