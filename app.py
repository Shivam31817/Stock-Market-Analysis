import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from io import StringIO
from prophet import Prophet

st.set_page_config(page_title="📈 Advanced Stock Market Analysis", layout="wide")
st.title("📊 Enhanced Stock Price Prediction App with Model Selection")

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
            future_preds = None
            pred_dates = None
            if prediction_model == "Linear Regression":
                X = data[['Days']]
                y = data['Close'].values.reshape(-1)
                model = LinearRegression()
                model.fit(X, y)
                future_X = pd.DataFrame({'Days': range(len(data), len(data) + future_days)})
                future_preds = model.predict(future_X)
                pred_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='B')[1:]

            elif prediction_model == "Prophet":
                df_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                model_prophet = Prophet()
                model_prophet.fit(df_prophet)
                future = model_prophet.make_future_dataframe(periods=future_days, freq='B')
                forecast = model_prophet.predict(future)
                future_preds = forecast['yhat'][-future_days:].values
                pred_dates = forecast['ds'][-future_days:].values

            # --- Debugging Outputs ---
            st.subheader(f"⚠️ Debugging Data for Plotting ({ticker})")
            st.write(f"Shape of data.index: {data.index.shape}, Type: {type(data.index)}")
            st.write(f"Shape of data['Close']: {data['Close'].shape}, Type: {type(data['Close'])}")
            if 'SMA_20' in data.columns:
                st.write(f"Shape of data['SMA_20']: {data['SMA_20'].shape}, Type: {type(data['SMA_20'])}")
            else:
                st.write("SMA_20 column not present.")
            if pred_dates is not None:
                st.write(f"Shape of pred_dates: {pred_dates.shape}, Type: {type(pred_dates)}")
            else:
                st.write("pred_dates is None.")
            if future_preds is not None:
                st.write(f"Shape of future_preds: {future_preds.shape}, Type: {type(future_preds)}")
            else:
                st.write("future_preds is None.")
            # --- End Debugging Outputs ---

            # Plotting
            if not data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name='Actual'))
                if 'SMA_20' in data.columns and not data['SMA_20'].isnull().all():
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA (20 days)'))
                if pred_dates is not None and future_preds is not None and len(pred_dates) > 0 and len(future_preds) > 0:
                    fig.add_trace(go.Scatter(x=pred_dates, y=future_preds, mode='lines+markers', name='Predicted'))
                elif pred_dates is None or future_preds is None:
                    st.warning(f"Prediction data is not available for {ticker}.")
                elif len(pred_dates) == 0 or len(future_preds) == 0:
                    st.warning(f"Prediction data has zero length for {ticker}.")


                st.plotly_chart(fig, use_container_width=True)

                # Predicted price for next day
                st.metric(f"📍 Next Day Predicted Price ({prediction_model})", f"${future_preds[0].item():.2f}" if future_preds is not None and future_preds.size > 0 else "N/A")

                # Show data with SMA
                st.subheader(f"📉 Recent Data with SMA")
                st.dataframe(data.tail(10))

                # CSV download
                combined_df = pd.concat([
                    data[['Close', 'SMA_20']].set_index(data.index),
                    pd.DataFrame({'Close': future_preds}, index=pred_dates) if pred_dates is not None and future_preds is not None and len(pred_dates) > 0 and len(future_preds) > 0 else pd.DataFrame()
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

            else:
                st.warning(f"No valid data to plot for {ticker} within the selected date range.")

        except Exception as e:
            st.error(f"Error for {ticker}: {e}")
