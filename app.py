from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Required so the HTML dashboard can call this API from the browser

# ── Load model ────────────────────────────────────────────────
model = pickle.load(open('rf_churn_model.pkl', 'rb'))

# ── Feature order — must match training exactly ───────────────
# ['Age', 'Gender', 'Location', 'Tenure_Months', 'Total_Spend',
#  'Num_Purchases', 'Last_Purchase_Days_Ago', 'Satisfaction_Score',
#  'Membership_Type', 'Complaints', 'Used_Discount', 'Avg_Monthly_Spend']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.json

        tenure      = float(d.get('Tenure_Months', 24))
        total_spend = float(d.get('Total_Spend', 50000))
        avg_monthly = total_spend / tenure if tenure > 0 else 0

        features = np.array([[
            float(d.get('Age', 35)),
            float(d.get('Gender', 0)),             # 0=Female, 1=Male
            float(d.get('Location', 0)),            # 0=North, 1=South, 2=West, 3=East
            tenure,
            total_spend,
            float(d.get('Num_Purchases', 10)),
            float(d.get('Last_Purchase_Days_Ago', 30)),
            float(d.get('Satisfaction_Score', 3)), # 1-5
            float(d.get('Membership_Type', 1)),    # 0=Gold, 1=Silver, 2=Bronze
            float(d.get('Complaints', 0)),
            float(d.get('Used_Discount', 0)),      # 0 or 1
            avg_monthly,
        ]])

        probability = float(model.predict_proba(features)[0][1])
        prediction  = int(model.predict(features)[0])

        return jsonify({
            'probability':   round(probability, 4),
            'churn':         prediction,
            'churn_percent': round(probability * 100, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':   'ok',
        'model':    'RandomForestClassifier',
        'accuracy': '99.5%',
        'features': 12
    })


if __name__ == '__main__':
    print("\n✅  LoyaltyPro ML API is running!")
    print("    POST http://localhost:5000/predict")
    print("    GET  http://localhost:5000/health\n")
    app.run(host='0.0.0.0', port=5001, debug=True)
