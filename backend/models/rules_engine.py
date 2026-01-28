"""
Basic Rules Engine - Rule-based fraud detection model
Version 1.0.0

Implements simple heuristic rules for fraud detection:
- High amount thresholds
- Transaction velocity patterns
- Geographic anomalies
- Time-based patterns
"""
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class RulesEngineModel:
    """Basic rule-based fraud detection engine"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.name = "Basic Rules Engine"
        
        # Rule thresholds
        self.high_amount_threshold = 1000.0  # $1000+
        self.very_high_amount_threshold = 5000.0  # $5000+
        self.velocity_window = 300  # 5 minutes in seconds
        self.velocity_threshold = 3  # 3 transactions in 5 min
        
        # Transaction history for velocity checks
        self.transaction_history: List[Dict] = []
        
        # Performance tracking
        self.total_predictions = 0
        self.fraud_detected = 0
        
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict fraud probability using rule-based logic
        
        Args:
            features: Array of shape (n_samples, 30) with [Time, V1-V28, Amount]
        
        Returns:
            Array of shape (n_samples, 2) with [prob_legit, prob_fraud]
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        probabilities = []
        
        for feature_vector in features:
            time_val = feature_vector[0]
            amount = feature_vector[-1]
            v_features = feature_vector[1:29]
            
            risk_score = self._calculate_risk_score(time_val, amount, v_features)
            
            # Convert risk score (0-100) to probability (0-1)
            fraud_prob = risk_score / 100.0
            legit_prob = 1.0 - fraud_prob
            
            probabilities.append([legit_prob, fraud_prob])
            self.total_predictions += 1
            
            if fraud_prob > 0.5:
                self.fraud_detected += 1
        
        return np.array(probabilities)
    
    def _calculate_risk_score(self, time: float, amount: float, v_features: np.ndarray) -> float:
        """
        Calculate risk score (0-100) based on multiple rules
        
        Returns:
            Risk score percentage (0-100)
        """
        risk_score = 0.0
        
        # Rule 1: High amount threshold (30 points)
        if amount > self.very_high_amount_threshold:
            risk_score += 30.0
        elif amount > self.high_amount_threshold:
            risk_score += 15.0
        
        # Rule 2: Unusual amount patterns (20 points)
        if amount > 0:
            # Round numbers are slightly suspicious
            if amount % 100 == 0:
                risk_score += 5.0
            # Very precise amounts might indicate automated fraud
            if amount != int(amount) and (amount * 100) % 1 == 0:
                risk_score += 10.0
        
        # Rule 3: Transaction velocity (25 points)
        velocity_risk = self._check_velocity(time)
        risk_score += velocity_risk
        
        # Rule 4: V-feature anomalies (25 points)
        # V-features are PCA components; extreme values indicate anomalies
        v_risk = self._check_v_features(v_features)
        risk_score += v_risk
        
        # Cap at 100
        risk_score = min(risk_score, 100.0)
        
        return risk_score
    
    def _check_velocity(self, current_time: float) -> float:
        """
        Check transaction velocity (transactions per time window)
        
        Returns:
            Risk points (0-25)
        """
        # Clean old transactions outside the window
        cutoff_time = current_time - self.velocity_window
        self.transaction_history = [
            tx for tx in self.transaction_history 
            if tx['time'] > cutoff_time
        ]
        
        # Add current transaction
        self.transaction_history.append({'time': current_time})
        
        # Check velocity
        recent_count = len(self.transaction_history)
        
        if recent_count >= self.velocity_threshold * 2:
            return 25.0  # Very high velocity
        elif recent_count >= self.velocity_threshold:
            return 15.0  # High velocity
        elif recent_count >= self.velocity_threshold - 1:
            return 5.0   # Moderate velocity
        
        return 0.0
    
    def _check_v_features(self, v_features: np.ndarray) -> float:
        """
        Check V-features for anomalies
        
        V-features are PCA components with mean ~0, std ~1
        Extreme values (> 3 std) are suspicious
        
        Returns:
            Risk points (0-25)
        """
        # Count extreme values (outliers)
        extreme_count = np.sum(np.abs(v_features) > 3.0)
        very_extreme_count = np.sum(np.abs(v_features) > 5.0)
        
        if very_extreme_count >= 3:
            return 25.0
        elif very_extreme_count >= 1:
            return 20.0
        elif extreme_count >= 5:
            return 15.0
        elif extreme_count >= 3:
            return 10.0
        elif extreme_count >= 1:
            return 5.0
        
        return 0.0
    
    def get_metrics(self) -> Dict:
        """Get current model performance metrics"""
        return {
            "name": self.name,
            "version": self.version,
            "total_predictions": self.total_predictions,
            "fraud_detected": self.fraud_detected,
            "fraud_rate": self.fraud_detected / max(self.total_predictions, 1),
            # Mock metrics for display (would be calculated from validation set)
            "accuracy": 0.85,
            "precision": 0.88,
            "recall": 0.92,
            "f1_score": 0.90,
            "false_positive_rate": 0.12,
            "false_negative_rate": 0.08,
        }
    
    def reset_history(self):
        """Reset transaction history (for testing)"""
        self.transaction_history = []
        self.total_predictions = 0
        self.fraud_detected = 0
