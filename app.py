from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Safe model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..loan_model.pkl")

model = joblib.load(model_path)

@app.route("/")
def home():
    return "Loan Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    # convert same as training
    df = pd.get_dummies(df)

    # align columns with training model
    model_features = model.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(df)[0]

    return jsonify({"Loan_Status": str(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
