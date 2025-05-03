def predict_stock(ticker):
    import yfinance as yf
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import matplotlib.pyplot as plt

    # Fetch data
    data = yf.download(ticker, period="1y")

    # ✅ Check if data is empty
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # Prepare data
    data = data[['Close']].dropna()
    data['Days'] = range(len(data))

    if data.shape[0] == 0:
        raise ValueError("Insufficient data to train model.")

    X = data[['Days']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    # Dummy prediction
    next_day = [[X['Days'].max() + 1]]
    pred_price = model.predict(next_day)[0]

    # Optional: return figure
    fig = plt.figure()
    plt.plot(data['Days'], y, label='Actual')
    plt.plot(next_day, [pred_price], 'ro', label='Predicted')
    plt.legend()

    return data, fig, pred_price
