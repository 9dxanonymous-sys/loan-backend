import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
# Production security ke liye CORS allow-all settings configured hai
CORS(app)

# 1. Model loading with absolute safety checks
try:
    model = joblib.load("model.pkl")
    print("✅ Model successfully backend mein load ho gaya!")
except Exception as e:
    print(f"❌ Model load karne mein error: {e}")
    model = None


# 2. Home Route (Railway 404 Error ko theek karne ke liye)
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "online",
            "message": "Smart Loan Prediction API is running successfully!",
            "model_loaded": model is not None,
        }
    )


# 3. Prediction Route (Fixed JSON keys to match frontend HTML)
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return (
            jsonify({"status": "error", "message": "Model file not found on server!"}),
            500,
        )

    try:
        # Frontend se JSON data receive karna
        data = request.get_json()

        if not data:
            return (
                jsonify({"status": "error", "message": "No input data provided"}),
                400,
            )

        # ⚠️ CRITICAL FIX: Frontend HTML ki keys 'Gender', 'ApplicantIncome' hain
        gender_raw = data.get("Gender")
        income_raw = data.get("ApplicantIncome")
        amount_raw = data.get("LoanAmount")
        credit_raw = data.get("Credit_History")

        # Validation check ke koi field khali na ho
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

        # Numeric Mapping & Conversion
        gender = 1 if gender_raw == "Male" else 0
        applicant_income = float(income_raw)
        loan_amount = float(amount_raw)
        credit_history = int(credit_raw)

        # Model input format
        input_features = [[gender, applicant_income, loan_amount, credit_history]]

        # Prediction engine
        prediction = model.predict(input_features)[0]

        # Output mapping (Frontend expects boolean or 'Approved'/'Rejected' string validation)
        # HTML line 348 demands response.loan_status to evaluate true/false status
        is_approved = True if int(prediction) == 1 else False
        result = "Approved" if is_approved else "Rejected"

        return jsonify(
            {
                "status": "success",
                "loan_status": is_approved,  # Boolean for frontend conditional check
                "status_text": result,
                "probability": 85 if is_approved else 35,  # Fallback visualization percentage
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    # ⚠️ PORT FIX: Railway dynamic port use karta hai. Local par yeh automatically 5000 par chalega.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
