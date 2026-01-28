import os
import uuid
from typing import List, Dict, Optional, Any

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  chromadb not found. Vector store disabled.")

class FraudEmbeddings:
    """Manages ChromaDB vector store for fraud patterns"""
    
    def __init__(self, persist_path: str = "backend/data/chroma_db"):
        self.persist_path = persist_path
        self.client = None
        self.collection = None
        
        if not CHROMA_AVAILABLE:
            return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(persist_path), exist_ok=True)
            
            # Using new ChromaDB client initialization
            self.client = chromadb.PersistentClient(path=persist_path)
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="fraud_patterns",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Vector Store initialized at {persist_path}")
            
            # Add some seed patterns if empty
            if self.collection.count() == 0:
                self._seed_patterns()
                
        except Exception as e:
            print(f"❌ ChromaDB Init Error: {e}")
            self.collection = None

    def _seed_patterns(self):
        """Add common fraud patterns to get started"""
        print("🌱 Seeding initial fraud patterns...")
        patterns = [
            "Rapid sequence of small transactions followed by a massive purchase at an electronics store.",
            "High amount transaction at 3:00 AM from an unusual foreign location.",
            "Multiple failed attempts followed by a successful medium-sized transaction at a gas station.",
            "Account takeover pattern: immediate large purchase at a luxury brand website after password change simulation.",
            "Gift card drain: series of small identical transactions at major retailers."
        ]
        
        # In a real app, we'd use actual embeddings. 
        # Here we'll rely on the client to generate them or use mock ones if GEMINI is unavailable.
        # For simplicity, we'll wait for real data from GeminiClient.
        pass

    def add_pattern(self, text: str, embedding: List[float], metadata: Optional[Dict] = None):
        """Add a fraud pattern to the vector DB"""
        if not self.collection or not embedding:
            return False
            
        try:
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata] if metadata else [{}],
                ids=[str(uuid.uuid4())]
            )
            return True
        except Exception as e:
            print(f"❌ Vector Store Add Error: {e}")
            return False

    def find_similar(self, query_embedding: List[float], n_results: int = 3) -> List[Dict[str, Any]]:
        """Find patterns similar to the query embedding"""
        if not self.collection or not query_embedding:
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            formatted = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    item = {
                        "description": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    }
                    formatted.append(item)
            return formatted
            
        except Exception as e:
            print(f"❌ Vector Search Error: {e}")
            return []
