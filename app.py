import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import BDay
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Stock Analyzer (Quantitative and Sentiments)")

# Request and receive user inputs
tkr = st.text_input("Enter stock ticker (e.g., AAPL, NVDA):").upper().strip()
st_dt = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
en_dt = st.date_input("End Date", value=datetime.today())
int_time = st.selectbox(
    "Select time interval:",
    options=[
        "1m", "2m", "5m", "15m", "30m", "60m", "90m",
        "1d", "5d", "1wk", "1mo", "3mo"
    ],
    index=6  # Defaults to "60m"
)

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
    return df

def arima_forecast(series, steps=30):
    model=ARIMA(series, order=(5,1,0)).fit() # 5 autoregressive terms (lags), the series is differenced 1 time to make it stationary, and has 0 moving average terms (error lags)
    forecast=model.forecast(steps=steps)
    return forecast 

def forecast_ets(series, steps=30):
    model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
    forecast = model.forecast(steps)
    return forecast

def forecast_prophet(series, steps=30):
    df_prophet = series.reset_index()
    df_prophet.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds')['yhat'][-steps:]

def forecast_prices(df, steps=30):
    series = df.set_index("Date")["Close"].asfreq('D').ffill()  # ensure daily frequency

    # Split into train/test for RMSE calculation
    train = series[:-steps]
    test = series[-steps:]

    models = {
        "ARIMA": arima_forecast,
        "ETS": forecast_ets,
        "Prophet": forecast_prophet
    }

    rmse_results = {}
    forecasts = {}

    for name, model_func in models.items():
        try:
            pred = model_func(train, steps=steps)
            pred.index = test.index  # align index for RMSE
            rmse = np.sqrt(mean_squared_error(test, pred))
            rmse_results[name] = rmse
            forecasts[name] = pred
        except Exception as e:
            rmse_results[name] = np.inf
            print(f"Model {name} failed: {e}")

    # Select best model
    best_model = min(rmse_results, key=rmse_results.get)
    best_forecast = forecasts[best_model]

    # Create forecast output DataFrame
    forecast_df = pd.DataFrame({
        "Date": best_forecast.index,
        "Forecast": best_forecast.values,
        "Lower CI": best_forecast.values * 0.98,  # mock confidence interval
        "Upper CI": best_forecast.values * 1.02
    })

    st.write(f"✅ Best model: **{best_model}** (RMSE: {rmse_results[best_model]:.2f})")

    return forecast_df


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
        plot_cols = ["Close", "VWAP", "TWAP"]
        macd_cols= ["MACD", "Signal_Line"]

        if df[plot_cols].notna().any().all():
            st.subheader("📉 Price, RSI & MACD Overview")
            st.line_chart(df.set_index("Date")[plot_cols])
            st.line_chart(df.set_index('Date')["RSI"])

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

# Forecast button
if st.button("📈 Forecast Future Prices"):
    df = get_stock_data(tkr, st_dt, en_dt, int_time)

    if df.empty:
        st.warning("⚠️ No data available to forecast. Please check ticker and date range.")
    elif int_time not in ["1d", "1wk"]:
        st.warning("⚠️ Forecasting is supported only for daily or weekly intervals. Please select '1d' or '1wk'.")
    else:
        try:
            df_forecast = forecast_prices(df.copy(), steps=30)
            st.subheader("📊 30-Day Price Forecast")
            st.dataframe(df_forecast)

            # Ensure 'Date' is a column (reset if necessary)
            if 'Date' not in df_forecast.columns:
                df_forecast = df_forecast.reset_index()

            # Plot forecast with confidence interval
            fig, ax = plt.subplots(figsize=(12, 5))

            # Plot forecast line
            ax.plot(df_forecast["Date"], df_forecast["Forecast"], label="Forecast", color="blue")

            # Plot confidence interval as shaded region
            ax.fill_between(df_forecast["Date"],
                            df_forecast["Lower CI"],
                            df_forecast["Upper CI"],
                            color='gray', alpha=0.3, label="Confidence Interval")

            # Labels and legend
            ax.set_title("30-Day Price Forecast with Confidence Interval")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)

            # Display in Streamlit
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
        
