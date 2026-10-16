from flask import Flask, render_template, request
import numpy as np
import pickle
import os
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)
with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)
transaction_map = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "CASH_IN": 3,
    "DEBIT": 4
}
location_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}
device_map = {
    "Mobile": 0,
    "Laptop": 1,
    "Desktop": 2,
    "Tablet": 3
}
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        transaction_id = float(request.form["transaction_id"])
        amount = float(request.form["amount"])
        transaction_type = transaction_map[
            request.form["transaction_type"]
        ]
        account_age_days = float(
            request.form["account_age_days"]
        )
        num_prev_transactions = float(
            request.form["num_prev_transactions"]
        )
        location_risk = location_map[
            request.form["location_risk"]
        ]
        device_type = device_map[
            request.form["device_type"]
        ]
        sample = np.array([[
            transaction_id,
            amount,
            transaction_type,
            account_age_days,
            num_prev_transactions,
            location_risk,
            device_type
        ]])
        sample = scaler.transform(sample.astype(np.float64))
        prediction = model.predict(sample)[0]
        if prediction == 1:
            prediction_text = "🚨 Fraudulent Transaction Detected"
            result_class = "fraud"
        else:
            prediction_text = "✅ Legitimate Transaction"
            result_class = "legitimate"
        return render_template(
            "index.html",
            prediction_text=prediction_text,
            result_class=result_class
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            result_class="error"
        )
if __name__ == "__main__":
    app.run(debug=True)