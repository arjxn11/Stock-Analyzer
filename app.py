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

# ----------------------------
# Data download helper
# ----------------------------
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

# ----------------------------
# Technical indicators
# ----------------------------
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
    ax.plot(df_macd.index, df_macd['MACD'], label='MACD', linewidth=1.5)
    ax.plot(df_macd.index, df_macd['Signal_Line'], label='Signal Line', linewidth=1.5)

    # Green for positive hist, red for negative
    colors = ['green' if val >= 0 else 'red' for val in df_macd['MACD_Hist']]
    ax.bar(df_macd.index, df_macd['MACD_Hist'], alpha=0.7, label='MACD Histogram', width=1, color=colors)

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

# ----------------------------
# FinBERT sentiment
# ----------------------------
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

        inputs = tokenizer(post.title, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        labels = model.config.id2label  # {0:'positive', 1:'negative', 2:'neutral'}
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

# ----------------------------
# Single-ticker analysis UI
# ----------------------------
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

        # Indicators
        df = calculate_vwap(df)
        df = calculate_twap(df)
        df = calculate_rsi(df)
        df = calculate_macd(df)

        # EPS and P/E Ratio
        trailingeps, forwardeps = calculate_eps(tkr)
        st.subheader("EPS and P/E ratio")
        st.write(f'**Trailing EPS:** {trailingeps}')
        st.write(f'**Forward EPS:** {forwardeps}')

        if trailingeps != "N/A" and isinstance(trailingeps, (float, int)) and trailingeps != 0:
            current_price = df['Close'].iloc[-1]
            pe_ratio = current_price / trailingeps
            st.write(f'**Current Price:** {current_price:.2f}')
            st.write(f'**Trailing P/E Ratio:** {pe_ratio:.2f}')
        else:
            st.warning("P/E Data unavailable")

        st.markdown(
            "If P/E ratio is high, typically over 25, that means investors expect significant growth from the company. "
            "Tech companies like Amazon and NVIDIA often have high P/E ratios due to rapid growth expectations. "
            "A low P/E ratio, typically 15 or lower, could suggest undervaluation (or lower growth expectations)."
        )

        # Preview
        st.subheader("📈 VWAP & TWAP")
        st.dataframe(df[['Date', 'Close', 'Volume', 'VWAP', 'TWAP']])

        # Debt-To-Equity Ratio 
        equity, debt = get_debt_equity(tkr)
        st.subheader("Debt to Equity Ratio")
        if equity:
            d_e = debt / equity if equity else None
            st.write(f'**Equity:** {equity:,.0f}')
            st.write(f'**Debt:** {debt:,.0f}')
            st.write(f'**Debt-to-Equity:** {d_e:.2f}')
        else:
            st.warning("Debt/Equity data not available for this ticker.")
        
        st.markdown(
            "A high D/E ratio indicates heavier reliance on debt (higher financial risk). "
            "A lower ratio suggests more equity financing (often more stability)."
        )

        # Plot line chart
        plot_cols = ["Close", "VWAP", "TWAP", "EMA_200"]
        macd_cols = ["MACD", "Signal_Line", "MACD_Hist"]

        if df[plot_cols].notna().any().all():
            st.subheader("📉 Price, RSI & MACD Overview")
            st.line_chart(df.set_index("Date")[plot_cols])
            st.line_chart(df.set_index('Date')["RSI"])

            st.markdown("""
            - **VWAP < TWAP**: More volume traded at lower prices.  
            - **EMA 200 + MACD**: Use together to spot trend entries/exits.  
            - **RSI**: >70 overbought (risk of pullback), <30 oversold (potential rebound). Divergences can flag reversals.
            """)
        else:
            st.warning("⚠️ One of Close/VWAP/TWAP/RSI/MACD/Signal_Line contains only NaNs — cannot plot.")

        if df[macd_cols].notna().any().all():
            st.subheader("📉 MACD & Signal Line")
            st.pyplot(plot_macd_dark(df))
        else:
            st.warning("⚠️ One or more of MACD components contain only NaNs — cannot plot.")

        st.markdown("MACD: when MACD crosses **above** Signal, potential **bullish** momentum; **below** suggests bearish momentum.")

        st.markdown("""
        ### 🔁 Trend Reversals and RSI Divergence
        - **Bearish divergence**: Price makes higher highs, RSI makes lower highs.  
        - **Bullish divergence**: Price makes lower lows, RSI makes higher lows.  
        Use with MACD/MA/volume for confirmation.
        """)

##################
# Price Forecast placeholder (single-ticker)
##################
if st.button("📊Price Forecast"):
    st.markdown("This Model is currently being constructed. We are working on getting this live and running for your use ASAP! Thank you for your patience!")

#########################
# Sentiment Analysis
#########################
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
            st.markdown("**Note:** Avg Positive Score=0.15 means the model sees a low probability of positive sentiment in posts.")

            st.write(sentiment_counts.rename("Count").to_frame())

            st.dataframe(
                sentiment_df.sort_values("Positive", ascending=False)[
                    ["Date", "Text", "Label", "Positive", "Negative", "Neutral"]
                ]
            )

            st.markdown("""
            - **Label**: Most confident class (highest probability)
            - **Positive/Negative/Neutral**: Class probabilities
            - Sort by `Positive`/`Negative` to surface strongest signals
            """)
        else:
            st.warning("No recent Reddit posts found related to this ticker.")
    else:
        st.warning("Enter a stock ticker to analyze Reddit sentiment.")

# ======================
# 📈 Portfolio Backtest + GBM (Black–Scholes) Forecast
# ======================
with st.form("portfolio_form"):
    st.subheader("Portfolio Monte Carlo Simulation + Backtest (GBM/Black–Scholes)")

    tickers_input = st.text_input("Enter tickers separated by commas (e.g. AAPL, MSFT, NVDA):").upper().strip()
    weights_input = st.text_input("Enter weights (comma-separated, must sum to 1):", "0.5,0.5")

    # GBM controls
    forecast_days   = st.number_input("Forecast horizon (trading days)", min_value=5, max_value=252, value=30, step=5)
    num_simulations = st.number_input("Number of simulations", min_value=100, max_value=5000, value=1000, step=100)
    drift_mode      = st.radio("Drift mode", ["Historical (real-world)", "Risk-neutral"], index=0, horizontal=True)
    risk_free_pct   = st.number_input("Risk-free rate (annual, %)", min_value=-5.0, max_value=15.0, value=4.0, step=0.25)
    use_correlation = st.checkbox("Use correlated shocks (Cholesky)", value=True)

    submitted = st.form_submit_button("💼 Run Portfolio Simulation")

if submitted:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) < 2:
        st.warning("Enter at least two tickers for a portfolio.")
    else:
        # Parse weights
        try:
            weights = [float(w.strip()) for w in weights_input.split(",")]
        except Exception:
            st.error("❌ Invalid weights format.")
            weights = []

        if len(weights) != len(tickers) or not np.isclose(sum(weights), 1):
            st.error("❌ Number of weights must match tickers and sum to 1.")
        else:
            # ---------- Download daily data ----------
            raw_data = yf.download(tickers, start=st_dt, end=en_dt, interval='1d', progress=False)
            if raw_data.empty:
                st.error("❌ No data downloaded. Check tickers, date range, or market holidays.")
            else:
                # Select price columns (preserve user order of tickers!)
                if isinstance(raw_data.columns, pd.MultiIndex):
                    cols_top = raw_data.columns.get_level_values(0).unique()
                    price_block = 'Adj Close' if 'Adj Close' in cols_top else 'Close'
                    data = raw_data[price_block].copy()
                else:
                    data = raw_data[['Adj Close']] if 'Adj Close' in raw_data.columns else raw_data[['Close']]

                # Keep only requested tickers, in the same order as user input
                available = [t for t in tickers if t in data.columns]
                missing = [t for t in tickers if t not in data.columns]
                if missing:
                    st.warning(f"Missing tickers (no price data): {', '.join(missing)}")
                if len(available) < 2:
                    st.error("Not enough valid tickers to form a portfolio.")
                else:
                    data = data[available].dropna()

                    # Align weights to the available columns order
                    if len(available) != len(weights):
                        w_map = dict(zip(tickers, weights))
                        weights = [w_map[t] for t in available]
                        if not np.isclose(sum(weights), 1):
                            st.error("After dropping missing tickers, weights no longer sum to 1. Please re-enter weights.")
                            st.stop()

                    # ---------- Backtest ----------
                    daily_returns = data.pct_change().dropna()
                    if daily_returns.empty:
                        st.error("❌ No returns after cleaning. Try a wider date range.")
                        st.stop()

                    weights_arr = np.array(weights)
                    portfolio_daily = (daily_returns * weights_arr).sum(axis=1)
                    cum_returns = (1 + portfolio_daily).cumprod()

                    # Equal-weight benchmark
                    eq_w = np.array([1/len(available)] * len(available))
                    benchmark_daily = (daily_returns * eq_w).sum(axis=1)
                    benchmark_cum = (1 + benchmark_daily).cumprod()

                    st.subheader("📊 Historical Portfolio Backtest")
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    ax1.plot(cum_returns.index, cum_returns, label="Your Portfolio", linewidth=2)
                    ax1.plot(benchmark_cum.index, benchmark_cum, label="Equal Weight Benchmark", linestyle="--")
                    ax1.set_xlabel("Date"); ax1.set_ylabel("Cumulative Return")
                    ax1.legend()
                    st.pyplot(fig1)

                    # ---------- GBM / Black–Scholes MC ----------
                    trading_days = 252
                    dt = 1.0 / trading_days
                    r = risk_free_pct / 100.0

                    price_df = data.copy().dropna()
                    log_ret = np.log(price_df / price_df.shift(1)).dropna()

                    mu_ann    = log_ret.mean() * trading_days                 # vector
                    Sigma_ann = log_ret.cov() * trading_days                 # covariance (annual)
                    sigma_ann = np.sqrt(np.diag(Sigma_ann))                  # vols

                    mu_vec = pd.Series(r, index=mu_ann.index) if drift_mode == "Risk-neutral" else mu_ann

                    S0 = price_df.iloc[-1].values
                    assets = list(price_df.columns)
                    n_assets = len(assets)

                    # Cholesky (with tiny jitter)
                    if use_correlation:
                        eps = 1e-10
                        try:
                            L = np.linalg.cholesky(Sigma_ann + eps * np.eye(n_assets))
                        except np.linalg.LinAlgError:
                            st.warning("Covariance not positive-definite; falling back to independent shocks.")
                            use_correlation = False

                    num_simulations = int(num_simulations)
                    sim_prices = np.zeros((forecast_days + 1, num_simulations, n_assets), dtype=float)
                    sim_prices[0, :, :] = S0

                    drift_term = (mu_vec.values - 0.5 * sigma_ann**2) * dt
                    Z = np.random.normal(size=(forecast_days, num_simulations, n_assets))

                    for t in range(1, forecast_days + 1):
                        if use_correlation:
                            shocks = (Z[t-1] @ L.T) * np.sqrt(dt)  # correlated annual → scale by sqrt(dt)
                        else:
                            shocks = Z[t-1] * (sigma_ann * np.sqrt(dt))
                        incr = drift_term + shocks
                        sim_prices[t] = sim_prices[t-1] * np.exp(incr)

                    # Portfolio paths (relative to S0)
                    rel_prices = sim_prices / S0
                    port_paths = np.tensordot(rel_prices, weights_arr, axes=([2], [0]))  # (time, sims)

                    mean_path = port_paths.mean(axis=1)
                    p5  = np.percentile(port_paths, 5, axis=1)
                    p95 = np.percentile(port_paths, 95, axis=1)

                    # Stitch normalized history (end = 1.0) with forecast
                    hist_index = cum_returns.index
                    hist_rel = (cum_returns / cum_returns.iloc[-1]).values
                    forecast_dates = pd.bdate_range(hist_index[-1] + pd.Timedelta(days=1), periods=forecast_days)

                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.plot(hist_index, hist_rel, linewidth=2, label="Historical Portfolio (normalized)")
                    ax2.plot(forecast_dates.insert(0, hist_index[-1]), np.insert(mean_path, 0, 1.0),
                             linewidth=2, label="Average GBM Forecast")
                    ax2.fill_between(forecast_dates.insert(0, hist_index[-1]),
                                     np.insert(p5, 0, 1.0),
                                     np.insert(p95, 0, 1.0),
                                     alpha=0.3, label="90% Confidence Interval")
                    ax2.set_title("Portfolio Value (GBM/Black–Scholes) — Correlated Monte Carlo")
                    ax2.set_xlabel("Date"); ax2.set_ylabel("Relative Value (end of history = 1.0)")
                    ax2.legend()
                    st.pyplot(fig2)

                    final_vals = port_paths[-1, :]
                    st.metric("Expected Final (rel.)", f"{final_vals.mean():.3f}")
                    st.metric("Best Case (95th %, rel.)", f"{np.percentile(final_vals,95):.3f}")
                    st.metric("Worst Case (5th %, rel.)", f"{np.percentile(final_vals,5):.3f}")
