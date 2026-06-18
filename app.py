```python
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load Model
try:
    model = joblib.load("model.pkl")
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Error:", e)
    model = None


@app.route("/")
def home():
    return jsonify({
        "project": "Smart Loan Approval System",
        "status": "Running",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        gender = 1 if data["Gender"] == "Male" else 0
        income = float(data["ApplicantIncome"])
        loan_amount = float(data["LoanAmount"])
        credit = int(data["Credit_History"])

        features = [[gender, income, loan_amount, credit]]

        prediction = model.predict(features)[0]

        return jsonify({
            "success": True,
            "loan_status": "Approved" if prediction == 1 else "Rejected",
            "loan_approved": bool(prediction)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
```
