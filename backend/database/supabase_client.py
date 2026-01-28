import os
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️  Supabase package not installed. Database will run in local mode.")

class SupabaseManager:
    """Manages connection to Supabase or fallback to local in-memory DB"""
    
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.client: Optional[Client] = None
        self.local_mode = False
        
        # In-memory storage for local mode
        self.local_db = {
            "transactions": [],
            "predictions": [],
            "fraud_patterns": []
        }

        if SUPABASE_AVAILABLE and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                print("✅ Connected to Supabase Successfully")
            except Exception as e:
                print(f"⚠️  Supabase connection failed: {e}. Switching to local mode.")
                self.local_mode = True
        else:
            if not self.url or not self.key:
                print("⚠️  Supabase credentials (SUPABASE_URL/SUPABASE_KEY) missing from .env.")
            print("ℹ️  Backend running in Local Database Mode (Mock DB).")
            self.local_mode = True

    def insert_transaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert transaction into database"""
        if self.local_mode:
            # Mock insert
            data['id'] = str(len(self.local_db['transactions']) + 1)
            data['created_at'] = datetime.now().isoformat()
            self.local_db['transactions'].append(data)
            return {"data": [data], "error": None}
        
        try:
            return self.client.table('transactions').insert(data).execute().model_dump()
        except Exception as e:
            print(f"❌ Supabase Insert Error: {e}")
            return {"data": [], "error": str(e)}

    def insert_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert prediction result"""
        if self.local_mode:
            data['id'] = str(len(self.local_db['predictions']) + 1)
            data['created_at'] = datetime.now().isoformat()
            self.local_db['predictions'].append(data)
            return {"data": [data], "error": None}
            
        try:
            return self.client.table('predictions').insert(data).execute().model_dump()
        except Exception as e:
            print(f"❌ Supabase Prediction Insert Error: {e}")
            return {"data": [], "error": str(e)}

    def get_recent_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent transactions"""
        if self.local_mode:
            # Return last N items reversed
            return self.local_db['transactions'][-limit:][::-1]
            
        try:
            response = self.client.table('transactions').select("*").order('created_at', desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            print(f"❌ Supabase Fetch Error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        if self.local_mode:
            total = len(self.local_db['transactions'])
            frauds = len([t for t in self.local_db['predictions'] if t.get('prediction') == True])
            blocked = len([t for t in self.local_db['predictions'] if t.get('prediction') == True]) # Simplified
            avg_risk = sum([float(t.get('fraud_probability', 0)) for t in self.local_db['predictions']]) / max(total, 1) * 100
            
            return {
                "total_transactions": total,
                "flagged_transactions": frauds,
                "blocked_transactions": blocked,
                "avg_risk_score": round(avg_risk, 2)
            }
            
        try:
            # Note: For free tier, selecting with count='exact' is best for small datasets
            total_res = self.client.table('transactions').select("*", count='exact').limit(1).execute()
            total = total_res.count if total_res.count is not None else 0
            
            frauds_res = self.client.table('predictions').select("*", count='exact').eq('prediction', True).limit(1).execute()
            frauds = frauds_res.count if frauds_res.count is not None else 0
            
            return {
                "total_transactions": total,
                "flagged_transactions": frauds,
                "blocked_transactions": frauds, # For now same as flagged
                "avg_risk_score": 0.0 # Calculate if needed
            }
        except Exception as e:
            print(f"❌ Supabase Stats Error: {e}")
            return {"total_transactions": 0, "flagged_transactions": 0, "blocked_transactions": 0, "avg_risk_score": 0.0}
