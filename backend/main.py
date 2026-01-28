"""
FraudGuard AI - FastAPI Backend

Real-time fraud detection API with WebSocket support for live monitoring
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from models.model_manager import ModelManager
from transaction_simulator import get_simulator
from database.supabase_client import SupabaseManager
from rag.gemini_client import GeminiClient
from rag.embeddings import FraudEmbeddings

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard AI API",
    description="Real-time fraud detection with multiple ML models",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()
simulator = get_simulator()

# Initialize external services
db = SupabaseManager()
gemini = GeminiClient()
vector_store = FraudEmbeddings()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.monitoring_active = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Statistics tracking
stats = {
    "total_transactions": 0,
    "flagged_transactions": 0,
    "blocked_transactions": 0,
    "total_risk_score": 0.0
}

# Recent transactions buffer
recent_transactions: List[Dict] = []
MAX_RECENT_TRANSACTIONS = 50


# Pydantic models for API
class Transaction(BaseModel):
    id: Optional[str] = None
    amount: float
    merchant: Optional[str] = None
    location: Optional[str] = None
    features: Optional[List[float]] = None
    risk_score: Optional[float] = None
    status: Optional[str] = None


class ExplanationRequest(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    location: str
    risk_score: float


class ModelActivation(BaseModel):
    model_id: str


class PredictionResponse(BaseModel):
    transaction_id: str
    predictions: Dict
    active_model: str
    blocking_decision: bool
    risk_score: float


# API Endpoints

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "FraudGuard AI",
        "version": "1.0.0",
        "models_loaded": True
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    metrics = model_manager.get_all_metrics()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "rules_engine": metrics["rules_engine"]["name"],
            "ml_model": f"{metrics['ml_model']['name']} (Loaded: {metrics['ml_model'].get('model_loaded', False)})",
            "dl_model": f"{metrics['dl_model']['name']} (Available: {metrics['dl_model'].get('tf_available', False)})",
        },
        "active_model": model_manager.active_model
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: Transaction):
    """
    Predict fraud for a single transaction
    
    Runs prediction through all three models
    """
    try:
        # Generate features if not provided
        if transaction.features is None:
            # Generate synthetic transaction
            synthetic_tx = simulator.generate_transaction()
            features = np.array(synthetic_tx["features"])
        else:
            features = np.array(transaction.features)
        
        # Ensure correct shape
        if len(features) != 30:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 30 features, got {len(features)}"
            )
        
        # Run prediction through all models
        predictions = model_manager.predict_all(features)
        
        # Create response
        tx_id = f"TX{stats['total_transactions'] + 1:06d}"
        active_pred = predictions[predictions["active_model"]]
        
        return {
            "transaction_id": tx_id,
            "predictions": predictions,
            "active_model": predictions["active_model"],
            "blocking_decision": predictions["blocking_decision"],
            "risk_score": active_pred["risk_score"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get current statistics (merges live session with DB if available)"""
    if db.local_mode:
        avg_risk = stats["total_risk_score"] / max(stats["total_transactions"], 1)
        return {
            "total_transactions": stats["total_transactions"],
            "flagged_transactions": stats["flagged_transactions"],
            "blocked_transactions": stats["blocked_transactions"],
            "avg_risk_score": round(avg_risk, 2)
        }
    else:
        # Fetch from Supabase
        return db.get_stats()


@app.post("/api/explain")
async def explain_transaction(request: ExplanationRequest):
    """
    Generate AI-powered explanation for a suspicious transaction
    Uses RAG to find similar historical patterns
    """
    try:
        # 1. Search for similar patterns in vector store if available
        # Description for embedding
        desc = f"Transaction of ${request.amount} at {request.merchant} in {request.location}"
        
        similar_cases = []
        if gemini.available:
            embedding = gemini.generate_embedding(desc)
            if embedding:
                similar_cases = vector_store.find_similar(embedding, n_results=2)
        
        # 2. Get AI explanation from Gemini
        tx_dict = request.model_dump()
        explanation = await gemini.explain_fraud(tx_dict, request.risk_score, similar_cases)
        
        return {
            "explanation": explanation,
            "similar_cases": similar_cases,
            "transaction_id": request.transaction_id
        }
    except Exception as e:
        print(f"❌ Explanation Error: {e}")
        return {
            "explanation": "Could not generate AI explanation at this time.",
            "error": str(e)
        }


@app.get("/api/transactions")
async def get_transactions(limit: int = 20):
    """Get recent transactions"""
    return recent_transactions[-limit:]


@app.get("/api/models")
async def get_models():
    """Get all model information"""
    return model_manager.get_all_metrics()


@app.get("/api/models/comparison")
async def get_model_comparison():
    """Get model comparison data for dashboard"""
    return model_manager.get_comparison_data()


@app.post("/api/models/activate")
async def activate_model(activation: ModelActivation):
    """
    Activate a specific model
    
    Body: { "model_id": "rules_engine" | "ml_model" | "dl_model" }
    """
    success = model_manager.set_active_model(activation.model_id)
    
    if success:
        return {
            "success": True,
            "active_model": model_manager.active_model,
            "message": f"Successfully activated {activation.model_id}"
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid model ID")


@app.websocket("/ws/transactions")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transaction streaming
    
    Client sends: { "action": "start" | "stop" }
    Server sends: transaction objects with predictions
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "start":
                # Start monitoring
                manager.monitoring_active = True
                await websocket.send_json({
                    "type": "status",
                    "message": "Monitoring started",
                    "monitoring": True
                })
                
                # Start transaction stream
                asyncio.create_task(transaction_stream())
                
            elif action == "stop":
                # Stop monitoring
                manager.monitoring_active = False
                await websocket.send_json({
                    "type": "status",
                    "message": "Monitoring stopped",
                    "monitoring": False
                })
            
            elif action == "ping":
                # Keep-alive
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        manager.monitoring_active = False


async def transaction_stream():
    """
    Background task that generates and broadcasts transactions
    """
    transaction_rate = float(os.getenv("TRANSACTION_RATE", "2.0"))
    interval = 1.0 / transaction_rate  # Time between transactions
    
    while manager.monitoring_active and len(manager.active_connections) > 0:
        try:
            # Generate transaction
            transaction = simulator.generate_transaction()
            
            # Get predictions from all models
            features = np.array(transaction["features"])
            predictions = model_manager.predict_all(features)
            
            # Determine status based on active model
            active_pred = predictions[predictions["active_model"]]
            risk_score = active_pred["risk_score"]
            
            if predictions["blocking_decision"]:
                status = "blocked"
                stats["blocked_transactions"] += 1
                stats["flagged_transactions"] += 1
            elif risk_score > 25:
                status = "flagged"
                stats["flagged_transactions"] += 1
            else:
                status = "approved"
            
            # Update stats
            stats["total_transactions"] += 1
            stats["total_risk_score"] += risk_score
            
            # Create transaction object for frontend
            tx_data = {
                "type": "transaction",
                "data": {
                    "id": transaction["id"],
                    "amount": transaction["amount"],
                    "merchant": transaction["merchant"],
                    "location": transaction["location"],
                    "timestamp": transaction["timestamp"],
                    "risk_score": round(risk_score, 1),
                    "status": status,
                    "predictions": {
                        "rules_engine": round(predictions["rules_engine"]["risk_score"], 1),
                        "ml_model": round(predictions["ml_model"]["risk_score"], 1),
                        "dl_model": round(predictions["dl_model"]["risk_score"], 1),
                    },
                    "active_model": predictions["active_model"],
                    "is_actual_fraud": transaction.get("is_actual_fraud", False),
                }
            }
            
            # Add to recent transactions
            recent_transactions.append(tx_data["data"])
            if len(recent_transactions) > MAX_RECENT_TRANSACTIONS:
                recent_transactions.pop(0)
            
            # Persist to Database in background
            if not db.local_mode:
                db.insert_transaction(tx_data["data"])
                db.insert_prediction({
                    "transaction_id": transaction["id"],
                    "model_name": predictions["active_model"],
                    "fraud_probability": risk_score / 100.0,
                    "prediction": status != "approved"
                })

            # Broadcast to all connected clients
            await manager.broadcast(tx_data)
            
            # Also send updated stats
            current_stats = await get_stats()
            await manager.broadcast({
                "type": "stats",
                "data": current_stats
            })
            
            # Wait before next transaction
            await asyncio.sleep(interval)
        
        except Exception as e:
            print(f"Error in transaction stream: {e}")
            await asyncio.sleep(1)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 60)
    print("🚀 FraudGuard AI Backend Starting...")
    print("=" * 60)
    
    # Reset stats
    stats["total_transactions"] = 0
    stats["flagged_transactions"] = 0
    stats["blocked_transactions"] = 0
    stats["total_risk_score"] = 0.0
    
    print("✅ Backend ready!")
    print(f"📊 Active Model: {model_manager.active_model}")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws/transactions")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
