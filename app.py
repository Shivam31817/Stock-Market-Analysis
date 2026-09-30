import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from io import StringIO
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- NewsAPI Key configuration ---
NEWS_API_KEY = "73b30eeff4514155a04655d5ad1e58b0"  

st.set_page_config(page_title="📈 Advanced Stock Market Analysis", layout="wide")
st.title("📊 Stock Price Analysis with ARIMA & Sentiment")

# --- Input Section ---
tickers_input = st.text_area("Enter Stock Ticker Symbols (comma-separated)", "AAPL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=365))
end_date = st.date_input("End Date")

future_days = st.slider("Predict how many future days?", 1, 10, 5)

if st.button("Run Analysis"):
    for ticker in tickers:
        st.markdown(f"---\n## 📈 Ticker: `{ticker}` - Using `ARIMA`")
        try:
            # Fetch data
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.warning(f"No data found for {ticker}")
                continue

            # Flatten Multi-Index columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data[['Close']].dropna().copy()
            
            # Standardize index frequency for ARIMA
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data = data.asfreq('B')  
            data['Close'] = data['Close'].ffill()  
            
            data['Days'] = range(len(data))

            # Calculate Simple Moving Average (SMA)
            data['SMA_20'] = data['Close'].rolling(window=20).mean()

            # --- ARIMA Prediction ---
            try:
                model_arima = ARIMA(data['Close'], order=(5, 1, 0))  
                model_arima_fit = model_arima.fit()
                forecast_result = model_arima_fit.get_forecast(steps=future_days)
                future_preds = forecast_result.predicted_mean.values
                pred_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='B')[1:]
            except Exception as e_arima:
                st.error(f"ARIMA Error for {ticker}: {e_arima}")
                future_preds = None
                pred_dates = None

            # Plotting with SMA and Predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name='Actual'))
            if 'SMA_20' in data.columns and not data['SMA_20'].isnull().all():
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA (20 days)'))
            if pred_dates is not None and future_preds is not None and len(pred_dates) > 0 and len(future_preds) > 0:
                fig.add_trace(go.Scatter(x=pred_dates, y=future_preds, mode='lines+markers', name='Predicted'))
            elif pred_dates is None or future_preds is None:
                st.warning(f"ARIMA prediction data is not available for {ticker}.")
            elif len(pred_dates) == 0 or len(future_preds) == 0:
                st.warning(f"ARIMA prediction data has zero length for {ticker}.")

            st.plotly_chart(fig, use_container_width=True)

            # Metric display extraction
            if future_preds is not None and future_preds.size > 0:
                next_day_val = float(future_preds.ravel()[0])
                st.metric("📍 Next Day Predicted Price (ARIMA)", f"\${next_day_val:.2f}")
            else:
                st.metric("📍 Next Day Predicted Price (ARIMA)", "N/A")

            # Show data with SMA
            st.subheader(f"📉 Recent Data with SMA")
            st.dataframe(data.tail(10))

            # CSV download compiler
            if pred_dates is not None and future_preds is not None and len(pred_dates) > 0 and len(future_preds) > 0:
                pred_df = pd.DataFrame({'Close': future_preds.ravel()}, index=pred_dates)
            else:
                pred_df = pd.DataFrame()
            
            combined_df = pd.concat([data[['Close', 'SMA_20']], pred_df])
            csv = combined_df.to_csv().encode('utf-8')
            st.download_button(
                label=f"⬇️ Download {ticker} Data with Prediction (ARIMA) as CSV",
                data=csv,
                file_name=f"{ticker}_predicted_data_arima.csv",
                mime='text/csv'
            )

            # Enhanced Stock Info
            st.subheader(f"📊 {ticker} - Stock Information")
            info = yf.Ticker(ticker).info
            st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A'):,.0f}")
            st.markdown(f"**PE Ratio (Trailing):** {info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else f"**PE Ratio (Trailing):** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**Dividend Yield:** {'{:.2%}'.format(info.get('dividendYield')) if isinstance(info.get('dividendYield'), float) else 'N/A'}")
            st.markdown(f"**Earnings Per Share (TTM):** {info.get('trailingEps', 'N/A'):.2f}" if isinstance(info.get('trailingEps'), (int, float)) else f"**Earnings Per Share (TTM):** {info.get('trailingEps', 'N/A')}")
            st.markdown(f"**Beta:** {info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else f"**Beta:** {info.get('beta', 'N/A')}")
            st.markdown(f"**Forward EPS:** {info.get('forwardEps', 'N/A'):.2f}" if isinstance(info.get('forwardEps'), (int, float)) else f"**Forward EPS:** {info.get('forwardEps', 'N/A')}")
            st.markdown(f"**Price to Book:** {info.get('priceToBook', 'N/A'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else f"**Price to Book:** {info.get('priceToBook', 'N/A')}")
            st.markdown(f"**Revenue Growth (YoY):** {'{:.2%}'.format(info.get('revenueGrowth')) if isinstance(info.get('revenueGrowth'), float) else 'N/A'}")

            # --- News Sentiment ---
            st.subheader(f"📰 {ticker} - News Sentiment")
            try:
                # Target endpoint with clean format properties string
                url = f"https://newsapi.org{ticker}&apiKey={NEWS_API_KEY}&sortBy=relevancy&pageSize=5"
                response = requests.get(url)
                response.raise_for_status()  
                news_data = response.json()
                sentiment_analyzer = SentimentIntensityAnalyzer()
                total_compound_score = 0
                
                if news_data.get("status") == "ok" and news_data.get("articles"):
                    for article in news_data["articles"]:
                        headline = article.get("title", "")
                        if headline:
                            vs = sentiment_analyzer.polarity_scores(headline)
                            total_compound_score += vs["compound"]
                    avg_sentiment = total_compound_score / len(news_data["articles"]) if news_data["articles"] else 0
                    
                    st.metric(label="Average Headline Sentiment Score", value=f"{avg_sentiment:.2f}")
                else:
                    st.info("Could not fetch news or no articles found.")
            except requests.exceptions.RequestException as e_news:
                st.error(f"Error fetching news: {e_news}")
            except Exception as e_sentiment:
                st.error(f"Error processing news sentiment: {e_sentiment}")

            # --- Analyst Ratings (Placeholder) ---
            st.subheader(f"📈 {ticker} - Analyst Ratings (Integration Placeholder)")
            st.info("Integration with an Analyst Ratings API would be added here. You would need to find a suitable API and implement the fetching and display logic.")

        except Exception as e:
            st.error(f"Error for {ticker}: {e}")
