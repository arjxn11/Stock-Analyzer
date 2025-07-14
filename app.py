import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer — VWAP & TWAP")

# -------------------- user input --------------------
ticker = st.text_input("Ticker (e.g. AAPL, NVDA):").upper().strip()
start  = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end    = st.date_input("End date",   value=datetime.today())

# -------------------- data loader -------------------
@st.cache_data
def load_data(tkr, s, e):
    if not tkr:
        return pd.DataFrame()

    # force multi-level then drop the ticker level
    df = yf.download(tkr, start=s, end=e, interval="1d",
                     group_by="ticker", auto_adjust=False)

    # df columns are MultiIndex like ('Close','NVDA') → keep level 0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().reset_index()          # 'Date' becomes a column
    return df

# -------------------- indicators --------------------
def add_vwap(df):
    vol = df["Volume"].replace(0, np.nan)
    df["VWAP"] = (df["Close"] * vol).cumsum() / vol.cumsum()
    return df

def add_twap(df):
    df["TWAP"] = df["Close"].expanding().mean()
    return df

# -------------------- app logic --------------------
if st.button("Analyze"):
    df = load_data(ticker, start, end)

    if df.empty:
        st.error("No data returned – check ticker or date range.")
        st.stop()

    df = add_twap(add_vwap(df))

    st.subheader("Raw data (first rows)")
    st.dataframe(df.head())

    st.subheader("VWAP & TWAP preview (last rows)")
    st.dataframe(df[["Date", "Close", "Volume", "VWAP", "TWAP"]].tail())

    st.subheader("Price chart")
    st.line_chart(df.set_index("Date")[["Close", "VWAP", "TWAP"]])
