import streamlit as st
import pandas as pd
from model import your_model_function  # Replace with actual function

st.set_page_config(page_title="Stock Market Analysis", layout="wide")

st.title("📈 Stock Market Analysis Dashboard")

# Upload or select a stock
ticker = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if st.button("Analyze"):
    try:
        df = your_model_function(ticker, start_date, end_date)  # Assuming it fetches and processes data
        st.success(f"Showing data for {ticker}")
        st.dataframe(df)

        # Plot
        st.line_chart(df["Close"])  # Replace with your actual columns
    except Exception as e:
        st.error(f"Error: {e}")

