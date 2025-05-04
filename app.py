import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from io import StringIO

st.set_page_config(page_title="📈 Advanced Stock Market Analysis", layout="wide")
st.title("📊 Advanced Stock Price Prediction App")

# --- Input Section ---
tickers_input = st.text_area("Enter Stock Ticker Symbols (comma-separated)", "AAPL, MSFT, GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

future_days = st.slider("Predict how many future days?", 1, 10, 5)

if st.button("Run Prediction"):
    for ticker in tickers:
        st.markdown(f"---\n## 📈 Ticker: `{ticker}`")

        try:
            # Fetch data
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.warning(f"No data found for {ticker}")
                continue

            data = data[['Close']].dropna().copy()
            data['Days'] = range(len(data))

            # Model training
            X = data[['Days']]
            y = data['Close']
            model = LinearRegression()
            model.fit(X, y)

            # Predict next N days
            future_X = pd.DataFrame({'Days': range(len(data), len(data) + future_days)})
            future_preds = model.predict(future_X)
            pred_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='B')[1:]

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=pred_dates, y=future_preds, mode='lines+markers', name='Predicted'))

            st.plotly_chart(fig, use_container_width=True)

            # Predicted price for next day
            st.metric("📍 Next Day Predicted Price", f"${future_preds[0]:.2f}")

            # Show data
            st.subheader("📉 Recent Data")
            st.dataframe(data.tail(10))

            # CSV download
            combined_df = pd.concat([
                data[['Close']],
                pd.DataFrame({'Close': future_preds}, index=pred_dates)
            ])
            csv = combined_df.to_csv().encode('utf-8')
            st.download_button(
                label="⬇️ Download Full Dataset as CSV",
                data=csv,
                file_name=f"{ticker}_predicted_data.csv",
                mime='text/csv'
            )

            # Stock info
            st.subheader("📊 Stock Info")
            info = yf.Ticker(ticker).info
            st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
            st.markdown(f"**PE Ratio (Trailing):** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**52 Week High / Low:** {info.get('fiftyTwoWeekHigh', 'N/A')} / {info.get('fiftyTwoWeekLow', 'N/A')}")

        except Exception as e:
            st.error(f"Error for {ticker}: {e}")

