import os
import logging

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google.generativeai not found. RAG disabled.")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.available = False
        
        if not GENAI_AVAILABLE:
             print("⚠️  Google Generative AI library not installed.")
             return

        if not self.api_key:
            print("⚠️  GEMINI_API_KEY not found. RAG features will be disabled.")
            return

        try:
            genai.configure(api_key=self.api_key)
            # Updated to use newer model as gemini-pro may be deprecated/unavailable
            self.model = genai.GenerativeModel('gemini-pro')
            self.available = True
            print("✅ Gemini API connected")
        except Exception as e:
            print(f"❌ Error configuring Gemini: {e}")

    def explain_fraud(self, transaction, prediction_prob, similar_frauds=None):
        """
        Generate a human-readable explanation for the fraud prediction.
        
        Args:
            transaction (dict): The transaction details (Amount, Time, etc.)
            prediction_prob (float): The probability of fraud (0.0 to 1.0)
            similar_frauds (list): List of similar historical fraud cases
        """
        if not self.available:
            return "AI explanation unavailable (Gemini Key missing)."

        # Format similar frauds for context
        context_str = ""
        if similar_frauds:
            context_str = "\nSimilar past patterns:\n"
            for i, f in enumerate(similar_frauds):
                context_str += f"- Case {i+1}: {f.get('description', 'Fraud pattern')}\n"

        prompt = f"""
        You are a fraud detection expert. Analyze this transaction:
        
        Transaction Details:
        - Amount: ${transaction.get('amount', 'N/A')}
        - Time (Sec): {transaction.get('time', 'N/A')}
        - ML Model Risk Score: {prediction_prob * 100:.1f}%
        
        {context_str}
        
        Task:
        1. Is this transaction suspicious? (Yes/No - based on score and usual patterns)
        2. Explain WHY in 2-3 short bullet points.
        3. Recommend an action (Approve / Flag for Review / Block).
        
        Keep the tone professional and concise.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating explanation: {e}"

    def generate_embedding(self, text):
        """Generate vector embedding for text using 'embedding-001'"""
        if not self.available:
            return None
            
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
                title="Fraud Pattern"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
