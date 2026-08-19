# LoyaltyPro — Customer Churn Prediction & Retention Dashboard

**[▶ Live demo](https://purvee25.github.io/loyaltypro-dashboard/)** — the trained Random Forest is exported to JSON and scored in your browser, so predictions are real with no server running.

A machine learning system that predicts which customers are likely to churn, served through a Flask REST API with an interactive HTML dashboard for retention teams.

## Results

| Metric | Score |
|---|---|
| Accuracy (held-out test set) | **99.5%** |
| Precision | 98.99% |
| Recall | 100% |
| F1 Score | 0.99 |

Model: **Random Forest** (100 trees, scikit-learn), trained on a 1,000-customer modeling dataset with an 80/20 train/test split, built from **2,000 customer profiles and 9,800+ transactions**.

> Note: scores are high because the dataset has strong signal — customers with repeated complaints churn at a much higher rate. Feature-importance analysis confirms the model learns from behavior, not label leakage.

### Top churn drivers (feature importance)

1. **Complaints** (0.53) — the dominant predictor; 3+ complaints almost always precede churn
2. **Satisfaction Score** (0.16)
3. **Purchase recency** — days since last purchase (0.16)

## Features used (12)

Age, Gender, Location, Tenure (months), Total Spend, Number of Purchases, Days Since Last Purchase, Satisfaction Score, Membership Type, Complaints, Used Discount, Average Monthly Spend

## Project structure

```
├── app.py                          # Flask API: /predict, /health
├── retrain.py                      # Trains the Random Forest and saves rf_churn_model.pkl
├── rf_churn_model.pkl              # Trained model
├── LoyaltyPro_Dashboard.html       # Interactive dashboard (calls the API from the browser)
├── cleaned_loyaltypro_dataset.csv  # 1,000-customer modeling dataset
├── customers.csv                   # 2,000 raw customer profiles
├── transactions.csv                # 9,800+ raw transactions
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# (Optional) retrain the model — prints test accuracy
python retrain.py

# Start the API
python app.py
```

Then open `LoyaltyPro_Dashboard.html` in a browser.

### API

```bash
# Health check
curl http://localhost:5000/health

# Churn prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 42, "Tenure_Months": 6, "Total_Spend": 12000, "Num_Purchases": 4, "Last_Purchase_Days_Ago": 75, "Satisfaction_Score": 2, "Complaints": 3, "Used_Discount": 1}'
```

Response:

```json
{ "probability": 0.97, "churn": 1, "churn_percent": 97.0 }
```

## Tech stack

Python · scikit-learn · Flask · pandas · NumPy · HTML/JS dashboard
