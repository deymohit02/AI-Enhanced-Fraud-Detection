import uuid
import os
import json

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not found. Vector store disabled.")

class FraudEmbeddings:
    def __init__(self, persist_path="processed_data/chroma_db"):
        """Initialize ChromaDB client"""
        if not CHROMA_AVAILABLE:
            self.client = None
            self.collection = None
            print("⚠️  ChromaDB not available.")
            return

        # Ensure directory exists
        if not os.path.exists(persist_path):
            os.makedirs(persist_path, exist_ok=True)
            
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="fraud_patterns",
            metadata={"hnsw:space": "cosine"} # Cosine similarity
        )
        print(f"✅ Vector Store initialized at {persist_path}")

    def add_pattern(self, text, embedding, metadata=None):
        """Add a fraud pattern to the vector DB"""
        if not CHROMA_AVAILABLE or not self.collection:
            return False

        if not embedding:
            return False
            
        if metadata is None:
            metadata = {}
            
        try:
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            return True
        except Exception as e:
            print(f"Error adding to vector store: {e}")
            return False

    def find_similar(self, query_embedding, n_results=3):
        """Find patterns similar to the query embedding"""
        if not CHROMA_AVAILABLE or not self.collection:
            return []

        if not query_embedding:
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
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
            print(f"Vector search error: {e}")
            return []

    def count(self):
        if not CHROMA_AVAILABLE or not self.collection:
            return 0
        return self.collection.count()
