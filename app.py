from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("../Model/loan_model.pkl")

# Load training columns (VERY IMPORTANT FIX)
df_template = pd.read_csv("../Dataset/train.csv")
df_template = df_template.dropna()
df_template["Loan_Status"] = df_template["Loan_Status"].map({"Y": 1, "N": 0})
df_template = pd.get_dummies(df_template.drop("Loan_Status", axis=1))
model_columns = df_template.columns


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # 1. Convert input to dataframe
    input_df = pd.DataFrame([data])

    # 2. One-hot encode like training
    input_df = pd.get_dummies(input_df)

    # 3. Align columns with training model
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # 4. Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # 5. Response
    return jsonify({
        "loan_status": int(prediction),
        "probability": round(float(probability) * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
