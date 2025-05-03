from flask import Flask, render_template, request
import yfinance as yf
from model import predict_stock

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    df, fig, pred_price = predict_stock(ticker)
    return render_template('result.html', ticker=ticker, fig=fig.to_html(), prediction=pred_price)

if __name__ == '__main__':
    app.run(debug=True)
