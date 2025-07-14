import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# Request and Receive User Input
st.title("Stock Analyzer")
ticker=st.text_input("Enter stock ticker (Eg: AAPL, NVDA): ")
start= st.date_input("start Date", value=pd.to_datetime('2023-01-01'))
end=st.date_input("End Date", value=datetime.today())


def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1h")
    df.dropna(inplace=True)
    return df

def calculate_vwap(df):
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def calculate_twap(df):
    df['TWAP'] = df['Close'].expanding().mean()
    return df

#update df
df=calculate_twap(df)
df=calculate_vwap(df)  
