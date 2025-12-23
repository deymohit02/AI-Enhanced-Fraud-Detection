import requests
import time
import json
import random
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "http://localhost:5000"

def test_health():
    print("Testing /health...")
    try:
        res = requests.get(f"{BASE_URL}/health")
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        return res.status_code == 200
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False

def test_predict():
    print("\nTesting /api/predict...")
    payload = {
        "amount": 150.00,
        "time": 3000,
        # We can optionally pass 'features' list if we want full control
        # "features": [...] 
    }
    
    res = requests.post(f"{BASE_URL}/api/predict", json=payload)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")

def test_transactions():
    print("\nTesting /api/transactions...")
    res = requests.get(f"{BASE_URL}/api/transactions")
    print(f"Status: {res.status_code}")
    # Print first 2 to avoid clutter
    data = res.json()
    print(f"Got {len(data)} transactions")
    if data:
        print(f"Sample: {data[0]}")

def test_stats():
    print("\nTesting /api/stats...")
    res = requests.get(f"{BASE_URL}/api/stats")
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")

if __name__ == "__main__":
    # Wait for server to potentially start if run immediately
    print("Waiting for API to be ready...")
    for _ in range(5):
        if test_health():
            break
        time.sleep(2)
        
    test_predict()
    test_transactions()
    test_stats()
