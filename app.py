import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer (VWAP & TWAP)")

# ---------------- User inputs ----------------
ticker = st.text_input("Ticker (e.g. AAPL, NVDA):").upper().strip()
start  = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end    = st.date_input("End date",   value=datetime.today())

# ---------------- Data retrieval ----------------
@st.cache_data
def get_data(tkr, st_dt, en_dt):
    if tkr == "":
        return pd.DataFrame()
    df = yf.download(tkr, start=st_dt, end=en_dt, interval="1d")   # <— NO group_by
    return df.dropna()

# ---------------- Indicator functions ----------------
def add_vwap(df):
    vol = df["Volume"].replace(0, np.nan)
    df["VWAP"] = (df["Close"] * vol).cumsum() / vol.cumsum()
    return df

def add_twap(df):
    df["TWAP"] = df["Close"].expanding().mean()
    return df

# ---------------- Main button ----------------
if st.button("Analyze"):
    df = get_data(ticker, start, end)

    if df.empty:
        st.error("No data — check ticker or date range.")
    else:
        df = df.reset_index()                       # 'Date' becomes a column
        df = add_vwap(add_twap(df))                 # add VWAP & TWAP

        st.subheader("Raw data (first rows)")
        st.dataframe(df.head())

        st.subheader("VWAP & TWAP preview (last rows)")
        st.dataframe(df[["Date", "Close", "Volume", "VWAP", "TWAP"]].tail())

        # ---------- Safe plotting ----------
        plot_cols = ["Close", "VWAP", "TWAP"]
        if df[plot_cols].notna().any().all():
            st.subheader("Price chart")
            st.line_chart(df.set_index("Date")[plot_cols])
        else:
            st.warning("One of Close/VWAP/TWAP is all NaN – cannot plot.")
