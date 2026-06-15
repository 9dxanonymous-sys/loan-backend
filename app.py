from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("Model/loan_model.pkl")

# Load training columns
df_template = pd.read_csv("Dataset/train.csv")
df_template = df_template.dropna()

df_template["Loan_Status"] = df_template["Loan_Status"].map({
    "Y": 1,
    "N": 0
})

df_template = pd.get_dummies(
    df_template.drop("Loan_Status", axis=1)
)

model_columns = df_template.columns


@app.route("/")
def home():
    return "Loan Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_df = pd.DataFrame([data])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "loan_status": int(prediction),
        "probability": round(float(probability) * 100, 2)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)