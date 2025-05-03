import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

def predict_stock(ticker):
    data = yf.download(ticker, period='1y')
    data = data.dropna()
    data['Days'] = range(len(data))

    X = data[['Days']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)
    next_day = [[len(data)]]
    pred_price = model.predict(next_day)[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Prices'))
    fig.add_trace(go.Scatter(x=[data.index[-1], data.index[-1]], y=[y.iloc[-1], pred_price],
                             mode='lines+markers', name='Prediction'))

    fig.update_layout(title=f'{ticker} Stock Price Forecast',
                      xaxis_title='Date', yaxis_title='Price')

    return data, fig, round(pred_price, 2)
