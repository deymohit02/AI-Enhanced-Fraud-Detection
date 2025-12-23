from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src directory to path to import database client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database.supabase_client import SupabaseManager
from src.rag.gemini_client import GeminiClient
from src.rag.embeddings import FraudEmbeddings

# Serve static files from the 'static' directory at root URL
app = Flask(__name__, static_folder='../../static', static_url_path='')
CORS(app)

# Initialize Services
db = SupabaseManager()
gemini = GeminiClient()
vector_store = FraudEmbeddings()
MODEL = None
SCALER = None

def load_artifacts():
    """Load ML model and scaler"""
    global MODEL, SCALER
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '../../models')
        
        with open(os.path.join(models_dir, 'xgboost.pkl'), 'rb') as f:
            MODEL = pickle.load(f)
            
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            SCALER = pickle.load(f)
            
        print("✅ Model and Scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# Load on startup
load_artifacts()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "database_connected": db.client is not None,
        "required_features": 30,
        "feature_names": ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    })

@app.route('/api/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        tx_id = data.get('transaction_id')
        
        # 1. Fetch transaction details (Mock or DB)
        # For simplicity, if we don't have DB fetch logic perfect yet, allow passing details directly
        # Or look up in local_db/Supabase
        
        # Simplification: Assume client sends {transaction_id, amount, ...} OR just explain this payload
        # Better: Look up in DB
        # For Demo: Just pass the data in request if ID lookup implementation is complex
        
        transaction = {
            "amount": data.get('amount', 0),
            "time": data.get('time', 0),
            "id": tx_id
        }
        
        # 2. Generate embedding for RAG
        # Description of transaction for semantic search
        query_text = f"Transaction amount ${transaction['amount']} at time {transaction['time']}"
        query_embedding = gemini.generate_embedding(query_text)
        
        # 3. Find similar historical frauds
        similar_frauds = []
        if query_embedding:
            similar_frauds = vector_store.find_similar(query_embedding, n_results=3)
            
        # 4. Generate AI Explanation
        prediction_prob = data.get('prob', 0.5) # Pass this from client ideally
        explanation = gemini.explain_fraud(transaction, prediction_prob, similar_frauds)
        
        return jsonify({
            "explanation": explanation,
            "similar_cases": similar_frauds,
            "transaction_id": tx_id
        })

    except Exception as e:
        print(f"Error in /explain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "error": "No data provided",
                "required_format": "{ 'features': [Time, V1, V2, ..., V28, Amount] }"
            }), 400
        
        features = data.get('features')
        
        # Validate input: REQUIRE all 30 features
        if not features:
            return jsonify({
                "error": "Missing 'features' field",
                "message": "This API requires ALL 30 features for accurate fraud detection",
                "required_features": [
                    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
                    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
                    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
                    "V28", "Amount"
                ],
                "format": "{ 'features': [time_value, v1, v2, ..., v28, amount_value] }",
                "note": "Partial data (only amount/time) leads to inaccurate predictions"
            }), 400
        
        # Ensure it's a list
        if isinstance(features, dict):
            features = list(features.values())
        
        # Ensure it's 2D array
        if not isinstance(features[0], list):
            features = [features]
        
        # Validate feature count
        if len(features[0]) != 30:
            return jsonify({
                "error": f"Invalid feature count: expected 30, got {len(features[0])}",
                "required": "[Time, V1-V28 (28 features), Amount] = 30 total features"
            }), 400

        # Convert to DataFrame with correct columns for Scaler
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df = pd.DataFrame(features, columns=columns)
        
        # Scale
        X_scaled = SCALER.transform(df)
        
        # Predict
        prob = float(MODEL.predict_proba(X_scaled)[0][1])
        is_fraud = bool(prob > 0.5) # Threshold
        
        result = {
            "fraud_probability": prob,
            "is_fraud": is_fraud,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Store in DB
        # 1. Transaction
        tx_data = {
            "amount": float(df['Amount'].iloc[0]),
            "features": {"raw": features[0]}, # Store input
            "is_fraud": None, # Actual label unknown yet
            "timestamp": pd.Timestamp.now().isoformat()
        }
        res_tx = db.insert_transaction(tx_data)
        
        # 2. Prediction
        if res_tx.get('data'):
             tx_id = res_tx['data'][0]['id']
             pred_data = {
                 "transaction_id": tx_id,
                 "model_name": "xgboost",
                 "fraud_probability": prob,
                 "prediction": is_fraud
             }
             db.insert_prediction(pred_data)
             result['transaction_id'] = tx_id
        
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    try:
        res = db.get_recent_transactions(limit=20)
        return jsonify(res.get('data', []))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = db.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
