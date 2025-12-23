import requests
import json
import sys

# URL of your local API
url = 'http://127.0.0.1:5000/api/predict'

# The specific payload with known fraud V-features and high amount
payload = {
  "features": [
    0,       # Time 
    -10.2817840384715, 6.30238478416897, -13.271718028752, 8.92511547634157, -9.97557831146449, -2.83251346361162, -12.7032526593299, 6.70684583868978, -7.07842395823848, -12.8056831898117, 6.78605830197451, -13.0642398936784, 1.1795245105938, -13.694873039573, 0.951479176238826, -10.9542857275047, -20.5835927904158, -7.51726213169106, 2.87235363605874, -0.247647527731929, 2.47941350799644, 0.366932842343897, 0.0428046480276363, 0.478278891799746, 0.157770817898023, 0.329900959238367, 0.163504340964582, -0.485552212695879,
    5000000  # Amount
  ]
}

print(f"🚀 Sending High-Probability Fraud Payload to {url}...")
print("-" * 50)

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ PREDICTION RECEIVED:")
        print(f"   Fraud Probability: {result['fraud_probability']:.6f} ({result['fraud_probability']*100:.4f}%)")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Transaction ID: {result.get('transaction_id')}")
        
    else:
        print(f"\n❌ Error {response.status_code}:")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("\n❌ Connection Refused.")
    print("   Make sure the Flask server is running: `python src/api/app.py`")
except Exception as e:
    print(f"\n❌ An error occurred: {e}")
