@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data.get("ticker", "").strip().upper()

        if not ticker:
            return jsonify({"error": "Ticker is required"}), 400

        df, fig, pred_price = predict_stock(ticker)
        return jsonify({
            "ticker": ticker,
            "prediction": round(pred_price, 2)
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
