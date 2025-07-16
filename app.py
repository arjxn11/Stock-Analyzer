import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import BDay

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
    return df

def forecast_prices(df, steps=30):
    df = df.set_index("Date").asfreq('B')  # Ensure business day frequency
    df['Volume'] = df['Volume'].fillna(method='ffill')
    df['Close'] = df['Close'].fillna(method='ffill')

    # Normalize volume to stabilize variance
    df['volume_scaled'] = df['Volume'] / 1e6

    # Fit SARIMAX model
    model = SARIMAX(df['Close'], exog=df[['volume_scaled']], order=(1, 1, 1))
    model_fit = model.fit(disp=False)

    # Forecast next 'steps' business days
    future_dates = [df.index[-1] + BDay(i) for i in range(1, steps + 1)]
    future_volume = df['volume_scaled'].iloc[-1]  # Use last known volume

    forecast = model_fit.get_forecast(steps=steps, exog=np.full((steps, 1), future_volume))
    pred = forecast.predicted_mean
    conf_int = forecast.conf_int()

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": pred.values,
        "Lower CI": conf_int.iloc[:, 0].values,
        "Upper CI": conf_int.iloc[:, 1].values
    })

    return forecast_df


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
        plot_cols = ["Close", "VWAP", "TWAP", "RSI"]
        macd_cols= ["MACD", "Signal_Line"]

        if df[plot_cols].notna().any().all():
            st.subheader("📉 Price, RSI & MACD Overview")
            st.line_chart(df.set_index("Date")[plot_cols])

            st.markdown("""
            - **VWAP < TWAP**: Indicates that more volume was traded at lower prices.  
            VWAP is often used by institutions to evaluate trading efficiency, while TWAP is used in execution algorithms to slice orders evenly over time.

            - **RSI (Relative Strength Index)**:  
            RSI is a momentum oscillator ranging from 0 to 100. Values **above 70** indicate *overbought* conditions (possible pullback), while **below 30** suggests *oversold* (potential rebound).  
            RSI is most effective in sideways markets and calculated over a 14-period window.  
            **Divergence** between RSI and price trends often signals potential **reversals**.""")
        else:
            st.warning("⚠️ One of Close/VWAP/TWAP/RSI/MACD/Signal_Line contains only NaNs — cannot plot.")
        if df[macd_cols].notna().any().all():
            st.subheader("📉 MACD & Signal Line")
            st.line_chart(df.set_index("Date")[macd_cols])
        else: 
            st.warning("⚠️ One of MACD/Signal_Line contains only NaNs — cannot plot.")
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
# ----------- PART 2: Forecast Button -----------
if st.button("📈 Forecast Future Prices"):
    df = get_stock_data(tkr, st_dt, en_dt)  # Initialize or refresh df
    if not df.empty:
        df_forecast = forecast_prices(df.copy(), steps=30)
        st.subheader("📊 30-Day Price Forecast")
        st.dataframe(df_forecast)

        chart_data = df_forecast.set_index("Date")
        st.line_chart(chart_data[['Forecast', 'Lower CI', 'Upper CI']])
        st.markdown("**Note:** Forecast is based on SARIMAX using historical closing prices and volume as exogenous input.")
    else:
        st.warning("⚠️ No data available to forecast. Please check ticker and date range.")
        
