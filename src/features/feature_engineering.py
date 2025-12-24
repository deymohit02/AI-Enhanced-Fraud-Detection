"""
Comprehensive Feature Engineering for Fraud Detection
Generates 100+ features from transaction data

Dataset Context:
- V1-V28: PCA-transformed features (confidential)
- Time: Seconds elapsed since first transaction
- Amount: Transaction amount

Generated features include:
- Time-based patterns (hour, day, velocity)
- Amount statistics and deviations
- Velocity features (transaction frequency)
- Statistical aggregations
- V-feature interactions and outliers
- Behavioral patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime


class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection.
    Generates 100+ features from transaction data.
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            cache_enabled: Whether to cache computed features for performance
        """
        self.cache_enabled = cache_enabled
        self._cache = {}
        self._scaler = None
        
    def generate_all_features(self, df: pd.DataFrame, is_batch: bool = True) -> pd.DataFrame:
        """
        Generate all engineered features from transaction data.
        
        Args:
            df: DataFrame with columns [Time, V1-V28, Amount]
            is_batch: If True, compute features requiring historical context
            
        Returns:
            DataFrame with original + engineered features
        """
        result = df.copy()
        
        # Sort by Time for sequential features
        if is_batch:
            result = result.sort_values('Time').reset_index(drop=True)
        
        # Generate feature groups
        result = self._add_time_features(result)
        result = self._add_amount_features(result, is_batch)
        
        if is_batch:
            result = self._add_velocity_features(result)
            result = self._add_statistical_features(result)
        
        result = self._add_v_feature_analysis(result)
        result = self._add_behavioral_features(result)
        
        return result
    
    # ====================
    # TIME-BASED FEATURES (~15 features)
    # ====================
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features"""
        
        # Convert seconds to hours
        hours = df['Time'] / 3600
        
        # Hour of day (0-23)
        df['time_hour'] = (hours % 24).astype(int)
        
        # Day since start
        df['time_day'] = (hours / 24).astype(int)
        
        # Day of week (assuming first transaction is Monday)
        df['time_day_of_week'] = (df['time_day'] % 7).astype(int)
        
        # Weekend indicator
        df['time_is_weekend'] = (df['time_day_of_week'] >= 5).astype(int)
        
        # Time of day binning
        df['time_is_night'] = ((df['time_hour'] >= 22) | (df['time_hour'] < 6)).astype(int)
        df['time_is_morning'] = ((df['time_hour'] >= 6) & (df['time_hour'] < 12)).astype(int)
        df['time_is_afternoon'] = ((df['time_hour'] >= 12) & (df['time_hour'] < 18)).astype(int)
        df['time_is_evening'] = ((df['time_hour'] >= 18) & (df['time_hour'] < 22)).astype(int)
        
        # Risk score based on time (night transactions are riskier)
        df['time_risk_score'] = df['time_is_night'] * 2 + df['time_is_evening'] * 0.5
        
        # Normalized time (0-1 within each day)
        df['time_normalized_daily'] = (hours % 24) / 24
        
        # Time since epoch (normalized)
        df['time_normalized_total'] = df['Time'] / df['Time'].max() if len(df) > 0 else 0
        
        return df
    
    # ====================
    # AMOUNT-BASED FEATURES (~20 features)
    # ====================
    
    def _add_amount_features(self, df: pd.DataFrame, is_batch: bool = True) -> pd.DataFrame:
        """Generate amount-based features"""
        
        # Log-transformed amount (handle zeros)
        df['amount_log'] = np.log1p(df['Amount'])
        
        # Square root transform
        df['amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Amount binning
        df['amount_is_micro'] = (df['Amount'] < 1).astype(int)
        df['amount_is_small'] = ((df['Amount'] >= 1) & (df['Amount'] < 10)).astype(int)
        df['amount_is_medium'] = ((df['Amount'] >= 10) & (df['Amount'] < 100)).astype(int)
        df['amount_is_large'] = ((df['Amount'] >= 100) & (df['Amount'] < 1000)).astype(int)
        df['amount_is_huge'] = (df['Amount'] >= 1000).astype(int)
        
        if is_batch and len(df) > 0:
            # Percentile-based features
            df['amount_percentile'] = df['Amount'].rank(pct=True)
            
            # Z-score
            mean_amt = df['Amount'].mean()
            std_amt = df['Amount'].std()
            df['amount_zscore'] = (df['Amount'] - mean_amt) / (std_amt + 1e-8)
            
            # Deviation from global average
            df['amount_dev_from_mean'] = df['Amount'] - mean_amt
            df['amount_dev_from_median'] = df['Amount'] - df['Amount'].median()
            
            # Amount to max/min ratios
            max_amt = df['Amount'].max()
            min_amt = df['Amount'].min()
            df['amount_to_max_ratio'] = df['Amount'] / (max_amt + 1e-8)
            df['amount_to_min_ratio'] = df['Amount'] / (min_amt + 1e-8)
            
            # Quantile indicators
            q25, q50, q75 = df['Amount'].quantile([0.25, 0.5, 0.75])
            df['amount_above_q75'] = (df['Amount'] > q75).astype(int)
            df['amount_below_q25'] = (df['Amount'] < q25).astype(int)
            
            # IQR-based outlier detection
            iqr = q75 - q25
            df['amount_is_outlier'] = ((df['Amount'] < (q25 - 1.5 * iqr)) | 
                                       (df['Amount'] > (q75 + 1.5 * iqr))).astype(int)
        else:
            # Single transaction - use simple features
            df['amount_percentile'] = 0.5
            df['amount_zscore'] = 0
            df['amount_dev_from_mean'] = 0
            df['amount_dev_from_median'] = 0
            df['amount_to_max_ratio'] = 0.5
            df['amount_to_min_ratio'] = 1.0
            df['amount_above_q75'] = 0
            df['amount_below_q25'] = 0
            df['amount_is_outlier'] = 0
        
        return df
    
    # ====================
    # VELOCITY FEATURES (~15 features)
    # ====================
    
    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate velocity-based features (requires sorted data)"""
        
        # Time differences
        df['time_delta'] = df['Time'].diff().fillna(0)
        
        # Transaction count in rolling windows
        for window in [60, 300, 900, 3600, 21600, 86400]:  # 1min, 5min, 15min, 1hr, 6hr, 24hr
            window_name = self._get_window_name(window)
            df[f'txn_count_{window_name}'] = self._count_in_window(df['Time'], window)
        
        # Amount sum in rolling windows
        for window in [3600, 21600, 86400]:  # 1hr, 6hr, 24hr
            window_name = self._get_window_name(window)
            df[f'amount_sum_{window_name}'] = self._sum_in_window(df['Time'], df['Amount'], window)
        
        # Transaction frequency (txn per hour)
        df['txn_frequency'] = df['txn_count_1hr'] / (1 + 1e-8)
        
        # Velocity acceleration (rate of change)
        df['velocity_accel_1hr'] = df['txn_count_1hr'].diff().fillna(0)
        
        # Burst detection (sudden spike in frequency)
        df['is_burst'] = (df['time_delta'] < 60).astype(int)  # Less than 1 minute
        
        return df
    
    def _count_in_window(self, time_series: pd.Series, window_seconds: int) -> pd.Series:
        """Count transactions within a rolling time window"""
        counts = []
        times = time_series.values
        
        for i, current_time in enumerate(times):
            count = np.sum((times[:i+1] >= current_time - window_seconds) & 
                          (times[:i+1] <= current_time))
            counts.append(count)
        
        return pd.Series(counts, index=time_series.index)
    
    def _sum_in_window(self, time_series: pd.Series, value_series: pd.Series, 
                       window_seconds: int) -> pd.Series:
        """Sum values within a rolling time window"""
        sums = []
        times = time_series.values
        values = value_series.values
        
        for i, current_time in enumerate(times):
            mask = (times[:i+1] >= current_time - window_seconds) & (times[:i+1] <= current_time)
            total = np.sum(values[:i+1][mask])
            sums.append(total)
        
        return pd.Series(sums, index=time_series.index)
    
    @staticmethod
    def _get_window_name(seconds: int) -> str:
        """Convert seconds to readable window name"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds//60}min"
        elif seconds < 86400:
            return f"{seconds//3600}hr"
        else:
            return f"{seconds//86400}day"
    
    # ====================
    # STATISTICAL FEATURES (~25 features)
    # ====================
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical aggregation features"""
        
        # Rolling statistics for Amount
        for window in [10, 50, 100]:
            df[f'amount_mean_{window}'] = df['Amount'].rolling(window, min_periods=1).mean()
            df[f'amount_std_{window}'] = df['Amount'].rolling(window, min_periods=1).std().fillna(0)
            df[f'amount_min_{window}'] = df['Amount'].rolling(window, min_periods=1).min()
            df[f'amount_max_{window}'] = df['Amount'].rolling(window, min_periods=1).max()
        
        # Exponential moving averages
        df['amount_ema_10'] = df['Amount'].ewm(span=10, adjust=False).mean()
        df['amount_ema_50'] = df['Amount'].ewm(span=50, adjust=False).mean()
        
        # Coefficient of variation
        df['amount_cv_10'] = (df['amount_std_10'] / (df['amount_mean_10'] + 1e-8))
        
        # Deviation from rolling mean
        df['amount_dev_rolling_10'] = df['Amount'] - df['amount_mean_10']
        df['amount_dev_rolling_50'] = df['Amount'] - df['amount_mean_50']
        
        return df
    
    # ====================
    # V-FEATURE ANALYSIS (~20 features)
    # ====================
    
    def _add_v_feature_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from V1-V28 PCA components"""
        
        # Select V columns
        v_cols = [f'V{i}' for i in range(1, 29)]
        v_data = df[v_cols].values
        
        # V-feature statistics
        df['v_mean'] = np.mean(v_data, axis=1)
        df['v_std'] = np.std(v_data, axis=1)
        df['v_min'] = np.min(v_data, axis=1)
        df['v_max'] = np.max(v_data, axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
        
        # Important V-feature cross products (based on typical fraud patterns)
        # These pairs often show correlation with fraud in literature
        df['v_cross_1_2'] = df['V1'] * df['V2']
        df['v_cross_3_7'] = df['V3'] * df['V7']
        df['v_cross_4_11'] = df['V4'] * df['V11']
        df['v_cross_12_14'] = df['V12'] * df['V14']
        df['v_cross_17_18'] = df['V17'] * df['V18']
        
        # V-feature ratios
        df['v_ratio_1_2'] = df['V1'] / (np.abs(df['V2']) + 1e-8)
        df['v_ratio_3_4'] = df['V3'] / (np.abs(df['V4']) + 1e-8)
        
        # Euclidean norm of V-vector
        df['v_norm'] = np.linalg.norm(v_data, axis=1)
        
        # Outlier detection in V-space (simplified Mahalanobis distance)
        if len(df) > 1:
            v_mean_global = np.mean(v_data, axis=0)
            v_std_global = np.std(v_data, axis=0) + 1e-8
            z_scores = (v_data - v_mean_global) / v_std_global
            df['v_mahalanobis'] = np.linalg.norm(z_scores, axis=1)
        else:
            df['v_mahalanobis'] = 0
        
        # Count of extreme V-values (|z-score| > 3)
        if len(df) > 1:
            v_zscores = stats.zscore(v_data, axis=0, nan_policy='omit')
            df['v_extreme_count'] = np.sum(np.abs(v_zscores) > 3, axis=1)
        else:
            df['v_extreme_count'] = 0
        
        return df
    
    # ====================
    # BEHAVIORAL FEATURES (~10 features)
    # ====================
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate behavioral pattern features"""
        
        # Interaction features
        df['amount_time_interaction'] = df['Amount'] * df['time_normalized_total']
        df['amount_hour_interaction'] = df['Amount'] * df['time_hour']
        
        # Pattern consistency (variance of recent amounts)
        if len(df) > 10:
            df['pattern_consistency'] = 1 / (df['Amount'].rolling(10, min_periods=1).std() + 1e-8)
        else:
            df['pattern_consistency'] = 1.0
        
        # Transaction regularity (time between transactions)
        if len(df) > 2:
            time_diffs = df['Time'].diff().fillna(0)
            df['time_regularity'] = 1 / (time_diffs.std() + 1e-8)
        else:
            df['time_regularity'] = 1.0
        
        # Anomaly score (composite of multiple factors)
        df['anomaly_score'] = (
            df['amount_is_outlier'] * 2 +
            df['time_is_night'] * 1.5 +
            (df['v_extreme_count'] > 5).astype(int) * 2 +
            df['is_burst'] * 1
        )
        
        # Risk indicator (multiple risk factors)
        df['risk_indicator'] = (
            (df['Amount'] > 500).astype(int) +
            df['time_is_night'] +
            (df['time_delta'] < 60).astype(int) +
            (df['txn_frequency'] > 5).astype(int) if 'txn_frequency' in df.columns else 0
        )
        
        return df
    
    def get_feature_names(self, include_original: bool = True) -> List[str]:
        """
        Get list of all feature names.
        
        Args:
            include_original: Whether to include V1-V28, Time, Amount
            
        Returns:
            List of feature names
        """
        engineered = [
            # Time features
            'time_hour', 'time_day', 'time_day_of_week', 'time_is_weekend',
            'time_is_night', 'time_is_morning', 'time_is_afternoon', 'time_is_evening',
            'time_risk_score', 'time_normalized_daily', 'time_normalized_total',
            
            # Amount features
            'amount_log', 'amount_sqrt',
            'amount_is_micro', 'amount_is_small', 'amount_is_medium', 
            'amount_is_large', 'amount_is_huge',
            'amount_percentile', 'amount_zscore',
            'amount_dev_from_mean', 'amount_dev_from_median',
            'amount_to_max_ratio', 'amount_to_min_ratio',
            'amount_above_q75', 'amount_below_q25', 'amount_is_outlier',
            
            # Velocity features
            'time_delta', 'txn_count_1min', 'txn_count_5min', 'txn_count_15min',
            'txn_count_1hr', 'txn_count_6hr', 'txn_count_24hr',
            'amount_sum_1hr', 'amount_sum_6hr', 'amount_sum_24hr',
            'txn_frequency', 'velocity_accel_1hr', 'is_burst',
            
            # Statistical features
            'amount_mean_10', 'amount_std_10', 'amount_min_10', 'amount_max_10',
            'amount_mean_50', 'amount_std_50', 'amount_min_50', 'amount_max_50',
            'amount_mean_100', 'amount_std_100', 'amount_min_100', 'amount_max_100',
            'amount_ema_10', 'amount_ema_50', 'amount_cv_10',
            'amount_dev_rolling_10', 'amount_dev_rolling_50',
            
            # V-feature analysis
            'v_mean', 'v_std', 'v_min', 'v_max', 'v_range',
            'v_cross_1_2', 'v_cross_3_7', 'v_cross_4_11', 'v_cross_12_14', 'v_cross_17_18',
            'v_ratio_1_2', 'v_ratio_3_4', 'v_norm', 'v_mahalanobis', 'v_extreme_count',
            
            # Behavioral features
            'amount_time_interaction', 'amount_hour_interaction',
            'pattern_consistency', 'time_regularity', 'anomaly_score', 'risk_indicator'
        ]
        
        if include_original:
            original = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            return original + engineered
        
        return engineered
    
    def get_feature_count(self) -> int:
        """Get total count of engineered features (excluding originals)"""
        return len(self.get_feature_names(include_original=False))
    
    def extract_features_for_api(self, amount: float, time: float = 0, 
                                  v_features: Optional[List[float]] = None) -> Dict:
        """
        Extract features for a single transaction (API use).
        
        Args:
            amount: Transaction amount
            time: Transaction time in seconds
            v_features: Optional V1-V28 features (if None, use zeros)
            
        Returns:
            Dictionary of features
        """
        # Create minimal DataFrame
        if v_features is None:
            v_features = [0.0] * 28
        
        data = {'Time': [time], 'Amount': [amount]}
        for i, v in enumerate(v_features, 1):
            data[f'V{i}'] = [v]
        
        df = pd.DataFrame(data)
        
        # Generate features (non-batch mode for single transaction)
        df_features = self.generate_all_features(df, is_batch=False)
        
        # Convert to dictionary
        return df_features.iloc[0].to_dict()
