import os
from typing import List, Dict, Optional, Any

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️  google.generativeai not found. AI features disabled.")

class GeminiClient:
    """Manages Google Gemini API interactions for fraud analysis and embeddings"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.available = False
        self.model = None
        
        if not GENAI_AVAILABLE:
            return

        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY":
            print("⚠️  GEMINI_API_KEY placeholder or missing. AI features disabled.")
            return

        try:
            genai.configure(api_key=self.api_key)
            # Using gemini-1.5-flash for faster responses
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.available = True
            print("✅ Gemini AI Provider Ready")
        except Exception as e:
            print(f"❌ Gemini Configuration Error: {e}")

    async def explain_fraud(self, transaction: Dict[str, Any], risk_score: float, 
                           similar_patterns: Optional[List[Dict]] = None) -> str:
        """Generate AI explanation for a fraudulent transaction"""
        if not self.available:
            return "AI explanation unavailable (API key missing or provider error)."

        # Format similar patterns context
        context_str = ""
        if similar_patterns:
            context_str = "\nKnown similar fraud patterns:\n"
            for i, p in enumerate(similar_patterns):
                context_str += f"- Pattern {i+1}: {p.get('description', 'Undisclosed pattern')}\n"

        prompt = f"""
        You are a Cyber-Security and Fraud Detection Expert. Analyze this transaction flagged by our AI systems.
        
        Transaction Details:
        - ID: {transaction.get('id', 'N/A')}
        - Amount: ${transaction.get('amount', 'N/A')}
        - Merchant: {transaction.get('merchant', 'N/A')}
        - Location: {transaction.get('location', 'N/A')}
        - Risk Score: {risk_score:.1f}%
        
        {context_str}
        
        Task:
        1. Classify the risk (Low/Medium/High/Critical).
        2. Identify 2-3 specific red flags based on these details.
        3. Provide a brief recommendation for the investigator.
        
        Keep your response concise, structured in bullet points, and professional.
        Max 4-5 total bullet points.
        """
        
        try:
            # Note: generate_content is synchronous normally, but we call it in async wrapper if needed
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Gemini Generation Error: {e}")
            return "Error generating AI explanation. Please try again later."

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding for text using 'text-embedding-004'"""
        if not self.available:
            return None
            
        try:
            # Using newer embedding model
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"❌ Gemini Embedding Error: {e}")
            return None
