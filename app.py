import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from io import StringIO
from prophet import Prophet  # Import Prophet

st.set_page_config(page_title="📈 Advanced Stock Market Analysis", layout="wide")
st.title("📊 Advanced Stock Price Prediction App with Model Selection")

# --- Input Section ---
tickers_input = st.text_area("Enter Stock Ticker Symbols (comma-separated)", "AAPL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

future_days = st.slider("Predict how many future days?", 1, 10, 5)

prediction_model = st.selectbox("Select Prediction Model:", ["Linear Regression", "Prophet"])

if st.button("Run Prediction"):
    for ticker in tickers:
        st.markdown(f"---\n## 📈 Ticker: `{ticker}` - Using `{prediction_model}`")
        try:
            # Fetch data
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.warning(f"No data found for {ticker}")
                continue

            data = data[['Close']].dropna().copy()
            data['Days'] = range(len(data))
            data['Date'] = data.index  # For Prophet

            # Calculate Simple Moving Average (SMA)
            data['SMA_20'] = data['Close'].rolling(window=20).mean()

            # Prediction Logic
            if prediction_model == "Linear Regression":
                # Model training
                X = data[['Days']]
                y = data['Close'].values.reshape(-1)
                model = LinearRegression()
                model.fit(X, y)

                # Predict next N days
                future_X = pd.DataFrame({'Days': range(len(data), len(data) + future_days)})
                future_preds = model.predict(future_X)
                pred_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='B')[1:]

            elif prediction_model == "Prophet":
                # Prepare data for Prophet
                df_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                model_prophet = Prophet()
                model_prophet.fit(df_prophet)

                # Create future dataframe
                future = model_prophet.make_future_dataframe(periods=future_days, freq='B')  # Business days
                forecast = model_prophet.predict(future)

                # Extract predictions and dates
                future_preds = forecast['yhat'][-future_days:].values
                pred_dates = forecast['ds'][-future_days:].values

            # Plotting with SMA and Predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA (20 days)'))
            fig.add_trace(go.Scatter(x=pred_dates, y=future_preds, mode='lines+markers', name='Predicted'))

            st.plotly_chart(fig, use_container_width=True)

            # Predicted price for next day
            st.metric(f"📍 Next Day Predicted Price ({prediction_model})", f"${future_preds[0].item():.2f}" if future_preds.size > 0 else "N/A")

            # Show data with SMA
            st.subheader(f"📉 Recent Data with SMA")
            st.dataframe(data.tail(10))

            # CSV download
            combined_df = pd.concat([
                data[['Close', 'SMA_20']].set_index(data.index),
                pd.DataFrame({'Close': future_preds}, index=pred_dates)
            ])
            csv = combined_df.to_csv().encode('utf-8')
            st.download_button(
                label=f"⬇️ Download {ticker} Data with Prediction ({prediction_model}) as CSV",
                data=csv,
                file_name=f"{ticker}_predicted_data_{prediction_model.replace(' ', '_')}.csv",
                mime='text/csv'
            )

            # Enhanced Stock Info
            st.subheader(f"📊 {ticker} - Stock Information")
            info = yf.Ticker(ticker).info
            st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A'):,.0f}")
            st.markdown(f"**PE Ratio (Trailing):** {info.get('trailingPE', 'N/A'):.2f}")
            st.markdown(f"**Dividend Yield:** {'{:.2%}'.format(info.get('dividendYield')) if isinstance(info.get('dividendYield'), float) else 'N/A'}")
            st.markdown(f"**52 Week High / Low:** {info.get('fiftyTwoWeekHigh', 'N/A'):.2f} / {info.get('fiftyTwoWeekLow', 'N/A'):.2f}")
            st.markdown(f"**Beta:** {info.get('beta', 'N/A'):.2f}")
            st.markdown(f"**Forward EPS:** {info.get('forwardEps', 'N/A'):.2f}")

        except Exception as e:
            st.error(f"Error for {ticker}: {e}")

