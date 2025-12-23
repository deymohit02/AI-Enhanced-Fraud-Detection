import pandas as pd
import pickle
import numpy as np

# Load model and scaler
try:
    with open('models/xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

print("Loading dataset...")
df = pd.read_csv('data/creditcard.csv')
frauds = df[df['Class'] == 1]

best_prob = 0
best_features = []
best_idx = 0

print(f"Scanning {len(frauds)} fraud cases for strongest signal...")

for idx, row in frauds.iterrows():
    # V1 to V28
    v_features = row.drop(['Time', 'Amount', 'Class']).values.tolist()
    
    # Test with the scenario that was failing: High Amount
    amount = 5000000 
    time = 0
    
    features = [time] + v_features + [amount]
    
    # Scale
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df_test = pd.DataFrame([features], columns=columns)
    X_scaled = scaler.transform(df_test)
    
    prob = model.predict_proba(X_scaled)[0][1]
    
    if prob > best_prob:
        best_prob = prob
        best_features = v_features
        best_idx = idx

import json
with open('best_fraud_vector.json', 'w') as f:
    json.dump(best_features, f)

print("-" * 30)
print(f"Best Case Index: {best_idx}")
print(f"Probability with 5B Amount: {best_prob:.5f}")
print("V-Features saved to best_fraud_vector.json")
