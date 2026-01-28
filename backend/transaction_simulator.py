"""
Transaction Simulator - Generates realistic credit card transactions

Creates synthetic transactions with:
- Realistic merchant names, locations, and amounts
- PCA-transformed features (V1-V28) matching credit card data
- Injected fraud patterns (10-15% fraud rate)
- Real-time streaming capability
"""
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from faker import Faker


class TransactionSimulator:
    """Generates realistic fraudulent and legitimate transactions"""
    
    def __init__(self, fraud_rate: float = 0.12, seed: int = 42):
        """
        Initialize transaction simulator
        
        Args:
            fraud_rate: Percentage of fraudulent transactions (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.fraud_rate = fraud_rate
        self.faker = Faker()
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        # Transaction counter
        self.transaction_count = 0
        self.start_time = datetime.now()
        
        # Merchant categories
        self.merchant_categories = [
            "Restaurant", "Shopping", "Gas Station", "Grocery",
            "Entertainment", "Travel", "Online", "Utility",
            "Healthcare", "Insurance", "Education", "Services"
        ]
    
    def generate_transaction(self) -> Dict:
        """
        Generate a single transaction
        
        Returns:
            Dict with transaction details and features
        """
        self.transaction_count += 1
        
        # Determine if this is fraud
        is_actual_fraud = random.random() < self.fraud_rate
        
        # Generate transaction details
        if is_actual_fraud:
            transaction = self._generate_fraud_transaction()
        else:
            transaction = self._generate_legitimate_transaction()
        
        # Add transaction metadata
        transaction["id"] = f"TX{self.transaction_count:06d}"
        transaction["timestamp"] = datetime.now().isoformat()
        transaction["is_actual_fraud"] = is_actual_fraud
        
        # Generate time feature (seconds since first transaction)
        elapsed = (datetime.now() - self.start_time).total_seconds()
        transaction["time"] = elapsed
        
        # Generate V-features (PCA components)
        v_features = self._generate_v_features(is_fraud=is_actual_fraud)
        transaction["v_features"] = v_features
        
        # Create feature vector for model: [Time, V1-V28, Amount]
        features = [transaction["time"]] + v_features + [transaction["amount"]]
        transaction["features"] = features
        
        return transaction
    
    def _generate_legitimate_transaction(self) -> Dict:
        """Generate a legitimate transaction"""
        # Legitimate transactions: smaller amounts, common merchants
        amount = self._sample_legitimate_amount()
        merchant_type = random.choice(self.merchant_categories)
        merchant_name = self._generate_merchant_name(merchant_type)
        location = self._generate_location()
        
        return {
            "amount": round(amount, 2),
            "merchant": merchant_name,
            "merchant_type": merchant_type,
            "location": location,
            "card_type": random.choice(["Visa", "Mastercard", "Amex"])
        }
    
    def _generate_fraud_transaction(self) -> Dict:
        """Generate a fraudulent transaction"""
        # Fraudulent transactions: unusual patterns
        fraud_type = random.choice(["high_amount", "unusual_location", "rapid_sequence", "odd_merchant"])
        
        if fraud_type == "high_amount":
            # Unusually high amount
            amount = np.random.uniform(1000, 10000)
            merchant = random.choice(["Electronics Store", "Jewelry", "High-End Fashion", "Unknown Merchant"])
        elif fraud_type == "unusual_location":
            # Foreign/unusual location
            amount = np.random.uniform(200, 2000)
            merchant = "International " + random.choice(self.merchant_categories)
        elif fraud_type == "rapid_sequence":
            # Part of rapid transaction sequence
            amount = np.random.uniform(100, 500)
            merchant = random.choice(["Gas Station", "ATM", "Online Store"])
        else:  # odd_merchant
            # Unusual merchant
            amount = np.random.uniform(300, 3000)
            merchant = "Unknown Merchant"
        
        location = self._generate_location(unusual=True)
        
        return {
            "amount": round(amount, 2),
            "merchant": merchant,
            "merchant_type": "Various",
            "location": location,
            "card_type": random.choice(["Visa", "Mastercard", "Amex"]),
            "fraud_pattern": fraud_type
        }
    
    def _sample_legitimate_amount(self) -> float:
        """Sample amount from realistic distribution"""
        # Most transactions are small, few are large (log-normal distribution)
        mu, sigma = 3.5, 1.2  # Mean and std of log(amount)
        amount = np.random.lognormal(mu, sigma)
        
        # Clip to reasonable range
        amount = np.clip(amount, 1.0, 2000.0)
        
        return amount
    
    def _generate_merchant_name(self, category: str) -> str:
        """Generate realistic merchant name"""
        if category == "Restaurant":
            return f"{self.faker.last_name()}'s {random.choice(['Diner', 'Cafe', 'Bistro', 'Grill'])}"
        elif category == "Shopping":
            return f"{self.faker.company()} {random.choice(['Store', 'Shop', 'Boutique'])}"
        elif category == "Gas Station":
            return random.choice(["Shell", "BP", "Exxon", "Chevron", "Mobil"])
        elif category == "Grocery":
            return random.choice(["Walmart", "Target", "Safeway", "Kroger", "Whole Foods"])
        elif category == "Online":
            return f"{self.faker.company()}.com"
        else:
            return self.faker.company()
    
    def _generate_location(self, unusual: bool = False) -> str:
        """Generate transaction location"""
        if unusual:
            # Foreign or unusual location
            return f"{self.faker.country()}"
        else:
            # Common US locations
            return f"{self.faker.city()}, {self.faker.state_abbr()}"
    
    def _generate_v_features(self, is_fraud: bool = False) -> List[float]:
        """
        Generate V1-V28 features (PCA components)
        
        These are similar to real credit card fraud datasets where
        V-features are PCA-transformed to protect privacy.
        
        Legitimate: mean~0, std~1, mostly normal distribution
        Fraud: More extreme values, different distribution
        """
        v_features = []
        
        for i in range(28):
            if is_fraud:
                # Fraud: higher chance of extreme values
                if random.random() < 0.3:  # 30% chance of extreme value
                    # Extreme outlier
                    value = np.random.normal(0, 4.0)
                else:
                    # Normal-ish but slightly offset
                    value = np.random.normal(0.5, 1.5)
            else:
                # Legitimate: mostly normal distribution around 0
                value = np.random.normal(0, 1.0)
            
            v_features.append(round(value, 6))
        
        return v_features
    
    def reset(self):
        """Reset transaction counter"""
        self.transaction_count = 0
        self.start_time = datetime.now()


# Singleton instance
_simulator = None

def get_simulator() -> TransactionSimulator:
    """Get singleton transaction simulator instance"""
    global _simulator
    if _simulator is None:
        _simulator = TransactionSimulator(fraud_rate=0.12)
    return _simulator
