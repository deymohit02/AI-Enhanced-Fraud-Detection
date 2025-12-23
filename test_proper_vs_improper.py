import requests
import pandas as pd
import json

BASE_URL = "http://localhost:5000"

print("=" * 60)
print("TESTING FRAUD DETECTION API WITH PROPER FEATURES")
print("=" * 60)

# Load real transaction data
df = pd.read_csv('data/creditcard.csv')

print("\n1. Testing with REAL NON-FRAUD transaction...")
print("-" * 60)
non_fraud_sample = df[df['Class'] == 0].iloc[0]
features_non_fraud = non_fraud_sample.drop('Class').tolist()

payload1 = {
    "features": features_non_fraud
}

response1 = requests.post(f"{BASE_URL}/api/predict", json=payload1)
result1 = response1.json()
print(f"Amount: ${non_fraud_sample['Amount']:.2f}")
print(f"Time: {non_fraud_sample['Time']}")
print(f"V-features: [non-zero values...]")
print(f"\n✅ RESULT: Fraud Probability = {result1['fraud_probability']:.4f}")
print(f"   Classification: {'🚨 FRAUD' if result1['is_fraud'] else '✅ LEGITIMATE'}")

print("\n\n2. Testing with REAL FRAUD transaction...")
print("-" * 60)
fraud_sample = df[df['Class'] == 1].iloc[0]
features_fraud = fraud_sample.drop('Class').tolist()

payload2 = {
    "features": features_fraud
}

response2 = requests.post(f"{BASE_URL}/api/predict", json=payload2)
result2 = response2.json()
print(f"Amount: ${fraud_sample['Amount']:.2f}")
print(f"Time: {fraud_sample['Time']}")
print(f"V-features: [non-zero values...]")
print(f"\n✅ RESULT: Fraud Probability = {result2['fraud_probability']:.4f}")
print(f"   Classification: {'🚨 FRAUD' if result2['is_fraud'] else '✅ LEGITIMATE'}")

print("\n\n3. Testing with INCOMPLETE DATA (current approach)...")
print("-" * 60)
payload3 = {
    "amount": 150.00,
    "time": 3000
}

response3 = requests.post(f"{BASE_URL}/api/predict", json=payload3)
result3 = response3.json()
print(f"Amount: $150.00")
print(f"Time: 3000")
print(f"V-features: [ALL ZEROS - UNREALISTIC!]")
print(f"\n❌ RESULT: Fraud Probability = {result3['fraud_probability']:.4f}")
print(f"   Classification: {'🚨 FRAUD' if result3['is_fraud'] else '✅ LEGITIMATE'}")
print(f"\n⚠️  This is flagged as fraud because all-zero V-features")
print(f"   never appear in training data!")

print("\n\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n📊 Key Findings:")
print(f"   • Real non-fraud: {result1['fraud_probability']:.2%} fraud probability")
print(f"   • Real fraud:     {result2['fraud_probability']:.2%} fraud probability")
print(f"   • Incomplete:     {result3['fraud_probability']:.2%} fraud probability")
print("\n💡 Conclusion:")
print("   The model needs FULL features (V1-V28) to work correctly.")
print("   Providing only 'amount' and 'time' fills V-features with zeros,")
print("   which the model treats as highly anomalous (fraudulent).")
