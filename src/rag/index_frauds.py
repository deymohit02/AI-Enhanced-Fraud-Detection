import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.gemini_client import GeminiClient
from src.rag.embeddings import FraudEmbeddings

def seed_fraud_patterns():
    print("Initializing RAG components...")
    gemini = GeminiClient()
    vector_store = FraudEmbeddings()
    
    if not gemini.available:
        print("❌ Gemini not available. Skipping indexing.")
        return

    print("Generating fraud patterns...")
    # In a real scenario, we'd load this from the dataset where Class=1
    # For this demo, we'll create some representative patterns
    patterns = [
        {
            "description": "High value transaction at unusual time (3 AM) with rounded amount.",
            "metadata": {"type": "timing_anomaly", "severity": "high"}
        },
        {
            "description": "Multiple small transactions in rapid succession from same IP.",
            "metadata": {"type": "velocity_check", "severity": "medium"}
        },
        {
            "description": "Transaction amount significantly higher than user's average spending history.",
            "metadata": {"type": "amount_anomaly", "severity": "high"}
        },
        {
            "description": "First time transaction with international merchant without travel notice.",
            "metadata": {"type": "location_anomaly", "severity": "medium"}
        },
        {
            "description": "Card testing pattern: small $1.00 auth followed by large purchase.",
            "metadata": {"type": "card_testing", "severity": "critical"}
        }
    ]
    
    count = 0
    for p in patterns:
        text = p['description']
        print(f"Indexing: {text[:50]}...")
        
        # Generate embedding
        embedding = gemini.generate_embedding(text)
        
        if embedding:
            # Add to vector DB
            success = vector_store.add_pattern(text, embedding, p['metadata'])
            if success:
                count += 1
        else:
            print("Failed to generate embedding (API limit or key issue?)")
            
    print(f"\n✅ Indexing complete. Added {count} patterns to Vector Store.")
    print(f"Total patterns in DB: {vector_store.count()}")

if __name__ == "__main__":
    seed_fraud_patterns()
