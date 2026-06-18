import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("model.pkl")
    print("✅ Model successfully loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "online",
            "message": "Smart Loan Prediction API is running successfully!",
            "model_loaded": model is not None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return (
            jsonify({"status": "error", "message": "Model file not found on server!"}),
            500,
        )

    try:
        data = request.get_json()

        if not data:
            return (
                jsonify({"status": "error", "message": "No input data provided"}),
                400,
            )

        gender_raw = data.get("Gender")
        income_raw = data.get("ApplicantIncome")
        amount_raw = data.get("LoanAmount")
        credit_raw = data.get("Credit_History")

        if (
            gender_raw is None
            or income_raw is None
            or amount_raw is None
            or credit_raw is None
        ):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Missing required fields in request parameters.",
                    }
                ),
                400,
            )

        gender = 1 if gender_raw == "Male" else 0
        applicant_income = float(income_raw)
        loan_amount = float(amount_raw)
        credit_history = int(credit_raw)

        input_features = [[gender, applicant_income, loan_amount, credit_history]]
        prediction = model.predict(input_features)[0]

        is_approved = True if int(prediction) == 1 else False
        result = "Approved" if is_approved else "Rejected"

        return jsonify(
            {
                "status": "success",
                "loan_status": is_approved,
                "status_text": result,
                "probability": 85 if is_approved else 35,
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
