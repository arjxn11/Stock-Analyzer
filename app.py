import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer (Quantitative and Sentiments)")

# Request and receive user inputs
tkr = st.text_input("Enter stock ticker (e.g., AAPL, NVDA):").upper().strip()
st_dt = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
en_dt = st.date_input("End Date", value=datetime.today())

# Get data
@st.cache_data
def get_stock_data(ticker, start, end):
    if ticker == "":
        return pd.DataFrame()
    df = yf.download(ticker, start=start, end=end, interval="1d")
    df.dropna(inplace=True)
    return df

# VWAP and TWAP calculations
def calculate_vwap(df):
    if 'Close' in df.columns and 'Volume' in df.columns:
        volume = df['Volume'].replace(0, np.nan)
        df['VWAP'] = (df['Close'] * volume).cumsum() / volume.cumsum()
    else:
        df['VWAP'] = np.nan
    return df

def calculate_twap(df):
    if 'Close' in df.columns:
        df['TWAP'] = df['Close'].expanding().mean()
    else:
        df['TWAP'] = np.nan
    return df

def calculate_eps(tkr):
    ticker = yf.Ticker(tkr)
    info = ticker.info

    trailingeps = info.get("trailingEps", "N/A")
    forwardeps = info.get("forwardEps", "N/A")
    return trailingeps, forwardeps




# Analysis
if st.button("Stock Analysis"):
    df = get_stock_data(tkr, st_dt, en_dt)

    if df.empty:
        st.error("❌ No data returned. Check the ticker symbol or date range.")
    else:
        df = df.reset_index()

        # --- Handle MultiIndex columns ---
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                st.warning("⚠️ Could not parse MultiIndex. Please try again.")
        else:
            df.columns = df.columns.str.strip()

        # Raw Data
        st.subheader("Raw Extracted Data")
        st.dataframe(df)

        # Adding VWAP and TWAP to the chart
        df = calculate_vwap(df)
        df = calculate_twap(df)

        #EPS and P/E Ratio
        trailingeps, forwardeps=calculate_eps(tkr)
        st.subheader("EPS and P/E ratio")
        st.write(f'**Trailing EPS:** {trailingeps}')
        st.write(f'**Forward EPS:** {forwardeps}')

        if trailingeps != "N/A" and isinstance(trailingeps, (float, int)) and trailingeps!=0:
            current_price=df['Close'].iloc[-1]
            pe_ratio=current_price/trailingeps
            st.write(f'**Current Price:** {current_price:.2f}')
            st.write(f'**Trailing P/E Ratio:** {pe_ratio:.2f}')
        else:
            st.warning("P/E Data unavailable")

        st.markdown("If P/E ratio is high, typically over 25, that means investors expect significant growth from the company and the investors are willing to pay more for each dollar of earnings, expecting company profits to rise in the future. \n Tech companies like Amazon and NVIDIA have high P/E ratios because of their rapid growth potential. \n A low P/E ratio, typically 15 or lower, could mean that the stock is undervalued, indicating an opportunity to open a position with that stock (or that it isn't expected grow much)")
        # Preview
        st.subheader("📈 VWAP & TWAP")
        st.dataframe(df[['Date', 'Close', 'Volume', 'VWAP', 'TWAP']])

        # Plot line chart
        plot_cols = ["Close", "VWAP", "TWAP"]
        if df[plot_cols].notna().any().all():
            st.subheader("📉 Price Chart")
            st.line_chart(df.set_index("Date")[plot_cols])
            st.markdown("If VWAP<TWAP, then we can infer that more volume was traded at lower prices. \n"" VWAP is often used by institutions to evaluate trading efficiency, while TWAP is often used for execution algorithms to slice orders evenly over time")
        else:
            st.warning("⚠️ One of Close/VWAP/TWAP is all NaN – cannot plot.")


