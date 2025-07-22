import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
from prophet import Prophet

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer (Quantitative, Sentiments, and Time Series Forecasting)")

# Request and receive user inputs
tkr = st.text_input("Enter stock ticker (e.g., AAPL, NVDA):").upper().strip()
st_dt = st.date_input("Start Date", value=pd.to_datetime("2024-06-01"))
en_dt = st.date_input("End Date", value=datetime.today())
int_time = st.selectbox(
    "Select time interval:",
    options=[
        "1m", "2m", "5m", "15m", "30m", "60m", "90m",
        "1d", "5d", "1wk", "1mo", "3mo"
    ],
    index=7  # Defaults to 1d
)
st.markdown("Note: Some tickers may not provide data for all intervals. Try 30m, 60m, 1d, 1w as a check in case you're unable to see data by-minute.")
# Get data
@st.cache_data
def get_stock_data(ticker, start, end, interval):
    if ticker == "":
        return pd.DataFrame()
    
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    df.dropna(inplace=True)

    # Normalize datetime index
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.rename_axis("Date").reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

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

def calculate_rsi(df, period=14):
    close = df['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df


def calculate_eps(tkr):
    ticker = yf.Ticker(tkr)
    info = ticker.info

    trailingeps = info.get("trailingEps", "N/A")
    forwardeps = info.get("forwardEps", "N/A")
    return trailingeps, forwardeps

def get_debt_equity(tkr: str):
    ticker = yf.Ticker(tkr)
    bs = ticker.balance_sheet

    try:
        # Try all common equity field variations
        equity_fields = ["Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]
        debt_fields = ["Total Debt", "Long Term Debt"]

        equity = None
        for eq in equity_fields:
            if eq in bs.index:
                equity = bs.loc[eq].iloc[0]
                break

        debt = None
        for d in debt_fields:
            if d in bs.index:
                debt = bs.loc[d].iloc[0]
                break

        if equity is not None and debt is not None:
            return equity, debt
        else:
            print("❌ Debt or Equity field not found.")
            return None, None

    except Exception as e:
        print(f"❌ Error retrieving data: {e}")
        return None, None

def calculate_macd(df, short=12, long=26, signal=9):
    df['EMA_short'] = df['Close'].ewm(span=short, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

def plot_macd_dark(df):
    macd_cols = ["MACD", "Signal_Line", "MACD_Hist"]
    df_macd = df.set_index("Date")[macd_cols].dropna()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Set dark background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot MACD and Signal Line
    ax.plot(df_macd.index, df_macd['MACD'], label='MACD', color='blue', linewidth=1.5)
    ax.plot(df_macd.index, df_macd['Signal_Line'], label='Signal Line', color='orange', linewidth=1.5)

    # Green for positive hist, red for negative
    colors = ['green' if val >= 0 else 'red' for val in df_macd['MACD_Hist']]
    ax.bar(df_macd.index, df_macd['MACD_Hist'], color=colors, alpha=0.7, label='MACD Histogram', width=1)

    # Remove grid and style text
    ax.grid(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

    ax.set_title("MACD Indicator")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")

    return fig

@st.cache_resource
def load_finbert():
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model


@st.cache_data(show_spinner=False)
def analyze_reddit_sentiment(tkr, days_back=7, posts=50):
    tokenizer, model = load_finbert()

    reddit = praw.Reddit(
        client_id=st.secrets["REDDIT_CLIENT_ID"],
        client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
        user_agent="stock-analyzer",
        check_for_async=False
    )
    reddit.read_only = True

    rows = []
    subs = reddit.subreddit("wallstreetbets+stocks+investing")
    query = tkr

    end_time = datetime.utcnow()
    start_time = end_time - pd.Timedelta(days=days_back)

    for post in subs.search(query, sort="new", limit=posts):
        created_time = datetime.utcfromtimestamp(post.created_utc)
        if created_time < start_time:
            continue

        # Tokenize and run through model
        inputs = tokenizer(post.title, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        labels = model.config.id2label  # Maps [0, 1, 2] to ['positive', 'negative', 'neutral']
        sentiment_scores = {labels[i]: round(p, 4) for i, p in enumerate(probs)}
        top_label = max(sentiment_scores, key=sentiment_scores.get)

        rows.append({
            "Date": created_time,
            "Text": post.title,
            "Label": top_label,
            "Positive": sentiment_scores["positive"],
            "Negative": sentiment_scores["negative"],
            "Neutral": sentiment_scores["neutral"]
        })

    return pd.DataFrame(rows)

# Forecast using Prophet (Facebook)
def forecast_arima(df, steps=7):
    df = df.copy().reset_index()
    df = df[['Date', 'Close']].dropna()
    df.set_index('Date', inplace=True)
    df = df.asfreq('B')  # business day frequency

    df['Close'] = df['Close'].interpolate()  # fill missing values

    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    forecast_res = model_fit.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    forecast_dates = pd.date_range(start=df.index[-1] + BDay(1), periods=steps, freq='B')

    return df['Close'], forecast, forecast_dates, conf_int

# Analysis
if st.button("Stock Analysis"):
    df = get_stock_data(tkr, st_dt, en_dt, int_time)

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
        df= calculate_rsi(df)
        df= calculate_macd(df)

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

        # Debt-To-Equity Ratio 
        equity, debt= get_debt_equity(tkr)
        st.subheader("Debt to Equity Ratio")
        if equity:
            d_e=debt/equity if equity else None
            st.write(f'**Equity:** {equity:,.0f}')
            st.write(f'**Debt:** {debt:,.0f}')
            st.write(f'**Debt-to-Equity:** {d_e:.2f}')
        else:
            st.warning("Debt/Equity data not available for this ticker.")
        
        st.markdown("A high D/E ratio indicates that a company is more reliant on debt to finance its operations, which indicates higher financial risk. \n A lower ratio indicates a company relies more on equity financing, suggesting lower financial risk and more stability. ")



        # Plot line chart
        plot_cols = ["Close", "VWAP", "TWAP", "EMA_200"]
        macd_cols= ["MACD", "Signal_Line", "MACD_Hist"]

        if df[plot_cols].notna().any().all():
            st.subheader("📉 Price, RSI & MACD Overview")
            st.line_chart(df.set_index("Date")[plot_cols])
            st.line_chart(df.set_index('Date')["RSI"])

            st.markdown("""
            - **VWAP < TWAP**: Indicates that more volume was traded at lower prices.  
            VWAP is often used by institutions to evaluate trading efficiency, while TWAP is used in execution algorithms to slice orders evenly over time.
                        
            - The 200 day EMA is used together with the MACD indicator to identify good opportunities to open long or short positions on a stock

            - **RSI (Relative Strength Index)**:  
            RSI is a momentum oscillator ranging from 0 to 100. Values **above 70** indicate *overbought* conditions (possible pullback), while **below 30** suggests *oversold* (potential rebound).  
            RSI is most effective in sideways markets and calculated over a 14-period window.  
            **Divergence** between RSI and price trends often signals potential **reversals**.""")
        else:
            st.warning("⚠️ One of Close/VWAP/TWAP/RSI/MACD/Signal_Line contains only NaNs — cannot plot.")
        if df[macd_cols].notna().any().all():
            st.subheader("📉 MACD & Signal Line")
            st.pyplot(plot_macd_dark(df))
        else:
            st.warning("⚠️ One or more of MACD components contain only NaNs — cannot plot.")
        st.markdown("MACD measures short vs long EMA difference. When MACD crosses **above** the Signal Line, it may signal **bullish momentum**. A **downward crossover** may suggest bearish sentiment.")
        # Explaination
        st.markdown("""
        ### 🔁 Trend Reversals and RSI Divergence
        A **reversal** refers to a change in the direction of a price trend—either from an uptrend to a downtrend or vice versa.  
        When RSI diverges from price action, it can signal weakening momentum:
        - **Bearish divergence**: Price makes new highs but RSI makes lower highs → Potential downtrend reversal.
        - **Bullish divergence**: Price makes new lows but RSI makes higher lows → Potential uptrend reversal.

        Divergence doesn't guarantee a reversal but can serve as an early warning. Confirm with other indicators like MACD, moving averages, or volume.
        """)

# Price Forecast
if st.button("📊Price Forecast"):
    df = get_stock_data(tkr, st_dt, en_dt, int_time)

    if df.empty or df['Close'].dropna().shape[0] < 30:
        st.warning("Not enough data to forecast.")
    else:
        actual, forecast, forecast_dates, conf_int = forecast_arima(df, steps=7)

        st.subheader("📈 ARIMA Forecast (Next 7 Business Days)")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual historical prices
        ax.plot(actual.index, actual, label="Historical Close", color='blue', linewidth=2)

        # Plot forecast
        ax.plot(forecast_dates, forecast, label="Forecast", color='orange', linestyle='--', marker='o', markersize=6, linewidth=2)

        # Plot confidence interval
        ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2, label="Confidence Interval")

        # Chart aesthetics
        ax.set_title(f"{tkr} — ARIMA Price Forecast", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        # Adding dark theme to make it more visible
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

        st.pyplot(fig)

        st.markdown("""
        - **Blue Line**: Historical closing prices  
        - **Orange Line**: ARIMA forecast with visible markers  
        - **Shaded Region**: 95% confidence interval  
        """)



# Sentiment Analysis

if st.button("📢 Analyze Reddit Sentiment"):
    if tkr:
        sentiment_df = analyze_reddit_sentiment(tkr)

        if not sentiment_df.empty:
            sentiment_counts = sentiment_df['Label'].value_counts()
            avg_positive = sentiment_df['Positive'].mean()
            avg_negative = sentiment_df['Negative'].mean()
            avg_neutral = sentiment_df['Neutral'].mean()

            st.subheader(f"🧠 FinBERT Sentiment Summary for r/{tkr}")
            st.metric(label="Avg Positive Score", value=f"{avg_positive:.3f}")
            st.metric(label="Avg Negative Score", value=f"{avg_negative:.3f}")
            st.metric(label="Avg Neutral Score", value=f"{avg_neutral:.3f}")
            st.markdown("**Note:** Avg Positive Score=0.15 means that the model sees a low chance of positive sentiment in posts.")

            st.write(sentiment_counts.rename("Count").to_frame())

            # Show sorted sentiment scores
            st.dataframe(
                sentiment_df.sort_values("Positive", ascending=False)[
                    ["Date", "Text", "Label", "Positive", "Negative", "Neutral"]
                ]
            )

            st.markdown("""
            - **Label**: Most confident sentiment classification (based on highest probability)
            - **Positive/Negative/Neutral**: Full probability scores for each sentiment class
            - You can sort the table by `Positive` or `Negative` to surface the strongest signals
            """)
        else:
            st.warning("No recent Reddit posts found related to this ticker.")
    else:
        st.warning("Enter a stock ticker to analyze Reddit sentiment.")
