import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np

st.title("Stock Analyzer")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, NVDA):").upper().strip()
start  = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end    = st.date_input("End Date",   value=datetime.today())

def get_stock_data(ticker, start, end):
    if ticker == "":
        return pd.DataFrame()
    df = yf.download(ticker, start=start, end=end, interval="1d")
    df.dropna(inplace=True)
    return df

def calculate_vwap(df):
    if 'Close' in df.columns and 'Volume' in df.columns:
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].replace(0, np.nan).cumsum()
    else:
        df['VWAP'] = np.nan
    return df

def calculate_twap(df):
    df['TWAP'] = df['Close'].expanding().mean()
    return df

if st.button("Analyze"):

    df = get_stock_data(ticker, start, end)

    if df.empty:
        st.error("No data returned. Check the ticker symbol or date range.")
    else:
        df = calculate_vwap(df)
        df = calculate_twap(df)

        # Print columns and non-null counts for debugging
        st.write("Columns available:", df.columns.tolist())
        st.write("Non-null counts:\n", df[['Close', 'VWAP', 'TWAP']].notna().sum())

        # Check both column presence and if they're not all NaNs
        wanted_cols = ['Close', 'VWAP', 'TWAP']
        if all(col in df.columns and df[col].notna().sum() > 0 for col in wanted_cols):
            st.line_chart(df[wanted_cols])
            st.write(df[['Close', 'Volume', 'VWAP', 'TWAP']].tail())
        else:
            st.warning("One or more of Close/VWAP/TWAP is missing or empty. Check volume or date range.")
