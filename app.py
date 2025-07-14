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
    df = yf.download(
        tkr,
        start=st_dt,
        end=en_dt,
        interval="1d",
        group_by="column",      # ensures flat column structure
        auto_adjust=False       # retain original prices and volume
    )
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

# --- Main Execution ---
if st.button("Analyze"):
    df = get_stock_data(ticker, start, end)

    if df.empty:
        st.error("❌ No data returned. Check the ticker symbol or date range.")
    else:
        df.columns = df.columns.astype(str).str.strip()  # Just in case
        df = calculate_vwap(df)
        df = calculate_twap(df)

        # --- Display Raw Data ---
        st.subheader("✅ Raw Extracted Data")
        st.dataframe(df.head())

        # --- Display VWAP & TWAP Preview ---
        st.subheader("📈 VWAP & TWAP Preview")
        preview_cols = ['Date', 'Close', 'Volume', 'VWAP', 'TWAP']
        available_cols = [col for col in preview_cols if col in df.columns]
        st.dataframe(df[available_cols].tail())

        # --- Plot Safely ---
        plot_cols = ["Close", "VWAP", "TWAP"]
        if all(col in df.columns and df[col].notna().any() for col in plot_cols):
            st.subheader("📉 Price Chart")
            st.line_chart(df.set_index("Date")[plot_cols])
        else:
            st.warning("One of Close/VWAP/TWAP is all NaN – cannot plot.")
