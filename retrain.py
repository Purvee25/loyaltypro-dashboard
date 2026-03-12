import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('cleaned_loyaltypro_dataset.csv')
FEATURES = ['Age','Gender','Location','Tenure_Months','Total_Spend','Num_Purchases','Last_Purchase_Days_Ago','Satisfaction_Score','Membership_Type','Complaints','Used_Discount','Avg_Monthly_Spend']
X, y = df[FEATURES], df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print('Accuracy:', round(accuracy_score(y_test, model.predict(X_test))*100, 2), '%')
pickle.dump(model, open('rf_churn_model.pkl', 'wb'))
print('Model saved!')
