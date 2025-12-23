import pickle
import numpy as np
import pandas as pd
import json

def load_artifacts():
    print("Loading artifacts...")
    with open('models/xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded")
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded")
    
    return model, scaler

def predict_sample(model, scaler):
    print("\n--- Running Test Prediction ---")
    
    # Create a random sample (30 features: Time + V1..V28 + Amount)
    # This mimics the input structure
    sample_legit = np.random.randn(1, 29) # 29 features (Time..V28) - No Amount yet
    
    # Wait, the scaler was fitted on 30 features (Time, V1..28, Amount)
    # Let's recreate a proper sample dataframe structure
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Generate random values
    data = []
    # Sample 1: Random noise (likely legit)
    row1 = [100.0] + list(np.random.normal(0, 1, 28)) + [50.0] 
    data.append(row1)
    
    df_sample = pd.DataFrame(data, columns=columns)
    
    print("Sample Data:")
    print(df_sample.iloc[0])
    
    # Scale
    print("\nScaling data...")
    X_scaled = scaler.transform(df_sample)
    
    # Predict
    print("Predicting...")
    # XGBoost expects specific input often, or numpy array
    prob = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]
    
    print(f"\nPrediction Result:")
    print(f"Fraud Probability: {prob:.4f}")
    print(f"Is Fraud: {bool(pred)}")
    
    if pred == 1:
        print("⚠️  FRAUD DETECTED!")
    else:
        print("✅  Transaction Legitimate")

if __name__ == "__main__":
    try:
        model, scaler = load_artifacts()
        predict_sample(model, scaler)
    except Exception as e:
        print(f"Error: {e}")
