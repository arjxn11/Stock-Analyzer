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

# Analysis
if st.button("Analysis (TWAP and VWAP)"):
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


@st.cache_data(show_spinner=False)  # cache the expensive API calls
def analyze_reddit_sentiment(tkr, posts=30, top_comments=10):
    reddit = praw.Reddit(
        client_id     = st.secrets["X09eMbi95NcVzuWk3xXdjg"],
        client_secret = st.secrets["AUXw94wqBL0sNWUnug6-W-4ti4LpKA"],
        user_agent    = "stock-analyzer",
        check_for_async=False,
        ratelimit_seconds=60,
    )
    reddit.read_only = True

    analyzer = SentimentIntensityAnalyzer()
    rows = []

    # fetch from a union of subs; r/stocks search is often throttled
    subs = reddit.subreddit("wallstreetbets+stocks")
    matches = subs.search(tkr, sort="new", limit=posts)

    for post in matches:
        rows.append({
            "Source":   "Title",
            "Text":     post.title,
            "Score":    (s := analyzer.polarity_scores(post.title))["compound"],
            "Sentiment": "Positive" if s > 0.05 else
                         "Negative" if s < -0.05 else "Neutral"
        })

        try:                     # comments may blow the 60-req/min allotment
            post.comments.replace_more(limit=0)
        except Exception:
            continue

        for c in post.comments[:top_comments]:
            rows.append({
                "Source": "Comment",
                "Text":   c.body,
                "Score":  (cs := analyzer.polarity_scores(c.body))["compound"],
                "Sentiment": "Positive" if cs > 0.05 else
                             "Negative" if cs < -0.05 else "Neutral"
            })

    return pd.DataFrame(rows)
