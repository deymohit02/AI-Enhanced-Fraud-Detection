"""
Test script for updated fraud detection API
Tests with real dataset transactions
"""

import requests
import pandas as pd
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("=" * 70)
    print("TESTING /health ENDPOINT")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResponse:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_with_real_transaction():
    """Test with a real non-fraud transaction"""
    print("\n" + "=" * 70)
    print("TEST 1: Real Non-Fraud Transaction")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv('data/creditcard.csv')
    sample = df[df['Class'] == 0].iloc[0]  # First non-fraud transaction
    features = sample.drop('Class').tolist()
    
    payload = {"features": features}
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ${sample['Amount']:.2f}")
    print(f"  Time: {sample['Time']}")
    print(f"  Actual Label: Non-Fraud")
    
    response = requests.post(f"{BASE_URL}/api/predict", json=payload)
    
    print(f"\nAPI Response:")
    if response.status_code == 200:
        result = response.json()
        print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"  Prediction: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
        print(f"  Transaction ID: {result.get('transaction_id', 'N/A')}")
        
        if result['fraud_probability'] < 0.5:
            print("\n✅ PASS: Correctly identified as legitimate")
        else:
            print("\n❌ FAIL: Incorrectly flagged as fraud")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

def test_with_fraud_transaction():
    """Test with a real fraud transaction"""
    print("\n" + "=" * 70)
    print("TEST 2: Real Fraud Transaction")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv('data/creditcard.csv')
    fraud_sample = df[df['Class'] == 1].iloc[0]  # First fraud transaction
    features = fraud_sample.drop('Class').tolist()
    
    payload = {"features": features}
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ${fraud_sample['Amount']:.2f}")
    print(f"  Time: {fraud_sample['Time']}")
    print(f"  Actual Label: Fraud")
    
    response = requests.post(f"{BASE_URL}/api/predict", json=payload)
    
    print(f"\nAPI Response:")
    if response.status_code == 200:
        result = response.json()
        print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"  Prediction: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
        print(f"  Transaction ID: {result.get('transaction_id', 'N/A')}")
        
        if result['fraud_probability'] > 0.5:
            print("\n✅ PASS: Correctly identified as fraud")
        else:
            print("\n❌ FAIL: Incorrectly classified as legitimate")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

def test_incomplete_data():
    """Test that incomplete data is properly rejected"""
    print("\n" + "=" * 70)
    print("TEST 3: Incomplete Data (Should Be Rejected)")
    print("=" * 70)
    
    payload = {
        "amount": 150.00,
        "time": 3000
    }
    
    print(f"\nPayload (missing V-features):")
    print(f"  Amount: $150.00")
    print(f"  Time: 3000")
    print(f"  V1-V28: NOT PROVIDED")
    
    response = requests.post(f"{BASE_URL}/api/predict", json=payload)
    
    print(f"\nAPI Response:")
    print(f"  Status Code: {response.status_code}")
    
    if response.status_code == 400:
        print("✅ PASS: API correctly rejected incomplete data")
        print(f"\nError Message:")
        error_data = response.json()
        print(f"  {error_data.get('error', 'Unknown error')}")
        if 'message' in error_data:
            print(f"  {error_data['message']}")
    else:
        print("❌ FAIL: API should reject incomplete data with 400 status")
        print(response.json())

def test_wrong_feature_count():
    """Test that wrong number of features is rejected"""
    print("\n" + "=" * 70)
    print("TEST 4: Wrong Feature Count (Should Be Rejected)")
    print("=" * 70)
    
    payload = {
        "features": [100, 200, 300]  # Only 3 features instead of 30
    }
    
    print(f"\nPayload: {len(payload['features'])} features (expected 30)")
    
    response = requests.post(f"{BASE_URL}/api/predict", json=payload)
    
    print(f"\nAPI Response:")
    print(f"  Status Code: {response.status_code}")
    
    if response.status_code == 400:
        print("✅ PASS: API correctly rejected wrong feature count")
        print(f"\nError Message:")
        print(f"  {response.json().get('error', 'Unknown error')}")
    else:
        print("❌ FAIL: API should reject wrong feature count with 400 status")

def test_batch_transactions():
    """Test multiple transactions"""
    print("\n" + "=" * 70)
    print("TEST 5: Batch Testing (10 transactions)")
    print("=" * 70)
    
    df = pd.read_csv('data/creditcard.csv')
    
    #Test 5 legitimate and 5 fraud
    legitimate_samples = df[df['Class'] == 0].sample(5, random_state=42)
    fraud_samples = df[df['Class'] == 1].sample(5, random_state=42)
    
    results = []
    
    print("\nLegitimate Transactions:")
    for idx, row in legitimate_samples.iterrows():
        features = row.drop('Class').tolist()
        response = requests.post(f"{BASE_URL}/api/predict", json={"features": features})
        if response.status_code == 200:
            result = response.json()
            correct = not result['is_fraud']
            symbol = "✅" if correct else "❌"
            print(f"  {symbol} Amount: ${row['Amount']:6.2f} | Prob: {result['fraud_probability']:.4f} | Pred: {'Fraud' if result['is_fraud'] else 'Legit'}")
            results.append(correct)
    
    print("\nFraud Transactions:")
    for idx, row in fraud_samples.iterrows():
        features = row.drop('Class').tolist()
        response = requests.post(f"{BASE_URL}/api/predict", json={"features": features})
        if response.status_code == 200:
            result = response.json()
            correct = result['is_fraud']
            symbol = "✅" if correct else "❌"
            print(f"  {symbol} Amount: ${row['Amount']:6.2f} | Prob: {result['fraud_probability']:.4f} | Pred: {'Fraud' if result['is_fraud'] else 'Legit'}")
            results.append(correct)
    
    accuracy = sum(results) / len(results) * 100
    print(f"\nBatch Accuracy: {accuracy:.1f}% ({sum(results)}/{len(results)} correct)")

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 FRAUD DETECTION API - COMPREHENSIVE TESTS")
    print("=" * 70)
    
    # Check if API is running
    try:
        if not test_health():
            print("\n❌ API health check failed. Is the server running?")
            print("Start it with: python src/api/app.py")
            return
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API at http://localhost:5000")
        print("Please start the server first: python src/api/app.py")
        return
    
    # Check if dataset exists
    try:
        df = pd.read_csv('data/creditcard.csv')
        print(f"\n✅ Dataset found: {len(df):,} transactions")
    except FileNotFoundError:
        print("\n❌ Dataset not found at data/creditcard.csv")
        print("Please run: python download_kaggle_dataset.py")
        return
    
    # Run tests
    test_with_real_transaction()
    test_with_fraud_transaction()
    test_incomplete_data()
    test_wrong_feature_count()
    test_batch_transactions()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
