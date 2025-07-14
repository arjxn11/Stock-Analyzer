import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

st.title("Stock Analyzer")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, NVDA):").upper().strip()
start  = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end    = st.date_input("End Date",   value=datetime.today())

def get_stock_data(ticker, start, end):
    if ticker == "":
        return pd.DataFrame()      # empty
    df = yf.download(ticker, start=start, end=end, interval="1h")
    df.dropna(inplace=True)
    return df

def calculate_vwap(df):
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
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

        # Make sure required columns exist
        wanted_cols = {'Close', 'VWAP', 'TWAP'}
        if wanted_cols.issubset(df.columns):
            st.line_chart(df[['Close', 'VWAP', 'TWAP']])
            st.write(df[['Close', 'Volume', 'VWAP', 'TWAP']].tail())
        else:
            st.warning("Close/VWAP/TWAP columns missing — cannot plot.")
