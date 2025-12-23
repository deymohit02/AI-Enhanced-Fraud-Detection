import os
import json
from datetime import datetime
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None

class SupabaseManager:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.client = None
        self.local_mode = False
        
        # In-memory storage for local testing
        self.local_db = {
            "transactions": [],
            "predictions": [],
            "fraud_patterns": []
        }

        if self.url and self.key and create_client:
            try:
                self.client = create_client(self.url, self.key)
                print("✅ Connected to Supabase")
            except Exception as e:
                print(f"⚠️  Supabase connection failed: {e}. Switching to local mode.")
                self.local_mode = True
        else:
            print("⚠️  Supabase credentials not found. Switching to local mode (Mock DB).")
            self.local_mode = True

    def insert_transaction(self, data):
        """Insert transaction into database"""
        if self.local_mode:
            # Mock insert
            data['id'] = str(len(self.local_db['transactions']) + 1)
            data['created_at'] = datetime.now().isoformat()
            self.local_db['transactions'].append(data)
            return {"data": [data], "error": None}
        
        return self.client.table('transactions').insert(data).execute()

    def insert_prediction(self, data):
        """Insert prediction result"""
        if self.local_mode:
            data['id'] = str(len(self.local_db['predictions']) + 1)
            data['created_at'] = datetime.now().isoformat()
            self.local_db['predictions'].append(data)
            return {"data": [data], "error": None}
            
        return self.client.table('predictions').insert(data).execute()

    def get_recent_transactions(self, limit=50):
        """Get recent transactions"""
        if self.local_mode:
            # Return last N items reversed
            return {"data": self.local_db['transactions'][-limit:][::-1], "error": None}
            
        return self.client.table('transactions').select("*").order('created_at', desc=True).limit(limit).execute()

    def get_stats(self):
        """Get dashboard statistics"""
        if self.local_mode:
            total = len(self.local_db['transactions'])
            frauds = len([t for t in self.local_db['predictions'] if t.get('prediction') == True])
            fraud_rate = (frauds / total * 100) if total > 0 else 0
            return {
                "total_transactions": total,
                "fraud_count": frauds,
                "fraud_rate": round(fraud_rate, 2)
            }
            
        # Supabase implementation would use count queries
        # For simplicity in free tier, we might just fetch recent or keep counters
        # Here is a basic implementation fetching count (might be slow for huge DBs)
        total = self.client.table('transactions').select("*", count='exact').execute().count
        frauds = self.client.table('predictions').select("*", count='exact').eq('prediction', True).execute().count
        
        return {
            "total_transactions": total,
            "fraud_count": frauds,
            "fraud_rate": round((frauds/total * 100), 2) if total else 0
        }
