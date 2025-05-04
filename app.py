import streamlit as st
from model import predict_stock  # ✅ Correct function
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Market Analysis", layout="wide")
st.title("📊 Stock Price Prediction")

# User input
ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL")

if st.button("Predict"):
    with st.spinner("Fetching data and predicting..."):
        try:
            data, fig, pred_price = predict_stock(ticker)

            st.success(f"Prediction complete for {ticker}")
            st.subheader("📉 Historical Closing Prices")
            st.dataframe(data.tail(10))

            st.subheader("📈 Prediction Plot")
            st.pyplot(fig)

            st.subheader("💰 Predicted Closing Price for Next Day:")
            st.metric(label="Predicted Price", value=f"${pred_price:.item():.2f}")


        except Exception as e:
            st.error(f"Error: {e}")
