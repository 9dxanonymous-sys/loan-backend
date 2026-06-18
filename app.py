from flask import Flask, jsonify, request
from flask_cors import CORS  # Agar frontend alag port par chal raha ho
import joblib

app = Flask(__name__)
CORS(app)  # Frontend-Backend connection errors se bachne ke liye

# 1. Model ko load karein (jo abhi train kiya hai)
try:
    model = joblib.load("model.pkl")
    print("✅ Model successfully backend mein load ho gaya!")
except Exception as e:
    print(f"❌ Model load karne mein error: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 2. Frontend se JSON data receive karein
        data = request.get_json()
        
        # 3. Data Extraction & Mapping (Frontend to Model Alignment)
        # Frontend string 'Male'/'Female' bhejta hai, model ko 1 ya 0 chahiye
        gender = 1 if data.get("gender") == "Male" else 0
        applicant_income = float(data.get("applicant_income"))
        loan_amount = float(data.get("loan_amount"))
        credit_history = int(data.get("credit_history")) # 1 (Good) ya 0 (Bad)
        
        # 4. Model ke liye input array banana
        input_features = [[gender, applicant_income, loan_amount, credit_history]]
        
        # 5. Prediction karna
        prediction = model.predict(input_features)[0]
        
        # 6. Output mapping (1 = Approved, 0 = Rejected)
        result = "Approved" if prediction == 1 else "Rejected"
        
        return jsonify({
            "status": "success",
            "loan_status": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    # Project ko run karne ke liye
    app.run(debug=True, port=5000)
