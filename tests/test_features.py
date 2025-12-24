"""
Test suite for feature engineering module
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data"""
        np.random.seed(42)
        n = 100
        
        data = {
            'Time': np.sort(np.random.randint(0, 172800, n)),  # 2 days in seconds
            'Amount': np.random.exponential(100, n)
        }
        
        # Add V features
        for i in range(1, 29):
            data[f'V{i}'] = np.random.randn(n)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def single_transaction(self):
        """Create a single transaction"""
        data = {
            'Time': [3600],
            'Amount': [150.50]
        }
        for i in range(1, 29):
            data[f'V{i}'] = [np.random.randn()]
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer(cache_enabled=True)
    
    # ==================== BASIC TESTS ====================
    
    def test_initialization(self, engineer):
        """Test FeatureEngineer initialization"""
        assert engineer is not None
        assert engineer.cache_enabled == True
        assert engineer.get_feature_count() > 80  # Should have 100+ features
    
    def test_generate_all_features_batch(self, engineer, sample_data):
        """Test batch feature generation"""
        result = engineer.generate_all_features(sample_data, is_batch=True)
        
        # Check output is DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check row count preserved
        assert result.shape[0] == sample_data.shape[0]
        
        # Check features added
        assert result.shape[1] > sample_data.shape[1]
        
        # Check no NaN in critical features
        assert not result['amount_log'].isna().any()
        assert not result['time_hour'].isna().any()
    
    def test_generate_all_features_single(self, engineer, single_transaction):
        """Test single transaction feature generation"""
        result = engineer.generate_all_features(single_transaction, is_batch=False)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 1
        assert result.shape[1] > single_transaction.shape[1]
    
    # ==================== TIME FEATURES ====================
    
    def test_time_features(self, engineer, sample_data):
        """Test time-based feature generation"""
        result = engineer._add_time_features(sample_data.copy())
        
        # Check new columns created
        assert 'time_hour' in result.columns
        assert 'time_day' in result.columns
        assert 'time_is_weekend' in result.columns
        assert 'time_is_night' in result.columns
        
        # Validate ranges
        assert (result['time_hour'] >= 0).all()
        assert (result['time_hour'] <= 23).all()
        assert result['time_is_weekend'].isin([0, 1]).all()
    
    def test_time_binning(self, engineer):
        """Test time of day binning"""
        # Create data with specific hours
        data = pd.DataFrame({
            'Time': [1800, 18000, 43200, 72000],  # 0.5hr, 5hr, 12hr, 20hr
            'Amount': [100, 200, 300, 400]
        })
        for i in range(1, 29):
            data[f'V{i}'] = 0
        
        result = engineer._add_time_features(data)
        
        # Check hour calculations
        assert result.loc[0, 'time_hour'] == 0
        assert result.loc[1, 'time_hour'] == 5
        assert result.loc[2, 'time_hour'] == 12
        assert result.loc[3, 'time_hour'] == 20
        
        # Check time of day flags
        assert result.loc[1, 'time_is_morning'] == 1
        assert result.loc[2, 'time_is_afternoon'] == 1
        assert result.loc[3, 'time_is_evening'] == 1
    
    # ==================== AMOUNT FEATURES ====================
    
    def test_amount_features(self, engineer, sample_data):
        """Test amount-based features"""
        result = engineer._add_amount_features(sample_data.copy(), is_batch=True)
        
        # Check new columns
        assert 'amount_log' in result.columns
        assert 'amount_sqrt' in result.columns
        assert 'amount_zscore' in result.columns
        assert 'amount_percentile' in result.columns
        
        # Validate transformations
        assert (result['amount_log'] >= 0).all()
        assert (result['amount_sqrt'] >= 0).all()
        assert (result['amount_percentile'] >= 0).all()
        assert (result['amount_percentile'] <= 1).all()
    
    def test_amount_binning(self, engineer):
        """Test amount binning"""
        data = pd.DataFrame({
            'Time': [0, 0, 0, 0, 0],
            'Amount': [0.5, 5, 50, 500, 5000]
        })
        for i in range(1, 29):
            data[f'V{i}'] = 0
        
        result = engineer._add_amount_features(data, is_batch=True)
        
        assert result.loc[0, 'amount_is_micro'] == 1
        assert result.loc[1, 'amount_is_small'] == 1
        assert result.loc[2, 'amount_is_medium'] == 1
        assert result.loc[3, 'amount_is_large'] == 1
        assert result.loc[4, 'amount_is_huge'] == 1
    
    # ==================== VELOCITY FEATURES ====================
    
    def test_velocity_features(self, engineer, sample_data):
        """Test velocity-based features"""
        result = engineer._add_velocity_features(sample_data.copy())
        
        # Check velocity columns
        assert 'time_delta' in result.columns
        assert 'txn_count_1hr' in result.columns
        assert 'txn_frequency' in result.columns
        assert 'is_burst' in result.columns
        
        # Check counts are positive
        assert (result['txn_count_1hr'] >= 0).all()
        assert (result['txn_frequency'] >= 0).all()
    
    def test_count_in_window(self, engineer):
        """Test transaction counting in time windows"""
        # Create transactions at specific times
        times = pd.Series([0, 30, 60, 90, 3600])  # 0s, 30s, 1min, 1.5min, 1hr
        
        counts_1min = engineer._count_in_window(times, 60)
        
        # At time 0: count = 1
        assert counts_1min.iloc[0] == 1
        # At time 60: count = 3 (0s, 30s, 60s)
        assert counts_1min.iloc[2] == 3
        # At time 3600: count = 1 (only itself)
        assert counts_1min.iloc[4] == 1
    
    # ==================== V-FEATURE ANALYSIS ====================
    
    def test_v_feature_analysis(self, engineer, sample_data):
        """Test V-feature analysis"""
        result = engineer._add_v_feature_analysis(sample_data.copy())
        
        # Check V statistics
        assert 'v_mean' in result.columns
        assert 'v_std' in result.columns
        assert 'v_norm' in result.columns
        assert 'v_mahalanobis' in result.columns
        
        # Check cross products
        assert 'v_cross_1_2' in result.columns
        assert 'v_cross_3_7' in result.columns
        
        # Validate calculations
        assert (result['v_std'] >= 0).all()
        assert (result['v_norm'] >= 0).all()
    
    # ==================== BEHAVIORAL FEATURES ====================
    
    def test_behavioral_features(self, engineer, sample_data):
        """Test behavioral pattern features"""
        result = engineer._add_behavioral_features(sample_data.copy())
        
        assert 'anomaly_score' in result.columns
        assert 'risk_indicator' in result.columns
        assert 'pattern_consistency' in result.columns
        
        # Scores should be non-negative
        assert (result['anomaly_score'] >= 0).all()
        assert (result['risk_indicator'] >= 0).all()
    
    # ==================== API FEATURES ====================
    
    def test_extract_features_for_api(self, engineer):
        """Test API feature extraction"""
        features = engineer.extract_features_for_api(
            amount=250.75,
            time=7200,
            v_features=[0.5] * 28
        )
        
        # Check it's a dictionary
        assert isinstance(features, dict)
        
        # Check critical features present
        assert 'Amount' in features
        assert 'Time' in features
        assert 'amount_log' in features
        assert 'time_hour' in features
        
        # Check values make sense
        assert features['Amount'] == 250.75
        assert features['Time'] == 7200
        assert features['time_hour'] == 2  # 7200s = 2hr
    
    def test_extract_features_without_v(self, engineer):
        """Test API extraction without V features"""
        features = engineer.extract_features_for_api(
            amount=100.0,
            time=0
        )
        
        assert isinstance(features, dict)
        assert 'V1' in features
        assert features['V1'] == 0.0  # Should default to 0
    
    # ==================== UTILITY TESTS ====================
    
    def test_get_feature_names(self, engineer):
        """Test feature name retrieval"""
        names_with_original = engineer.get_feature_names(include_original=True)
        names_without_original = engineer.get_feature_names(include_original=False)
        
        # Check lengths
        assert len(names_with_original) > len(names_without_original)
        
        # Check originals present
        assert 'Time' in names_with_original
        assert 'Amount' in names_with_original
        assert 'V1' in names_with_original
        
        # Check originals not in engineered-only
        assert 'Time' not in names_without_original
        assert 'Amount' not in names_without_original
    
    def test_get_feature_count(self, engineer):
        """Test feature count"""
        count = engineer.get_feature_count()
        
        # Should have 80+ engineered features
        assert count >= 80
        assert count <= 120  # Sanity check
    
    # ==================== EDGE CASES ====================
    
    def test_zero_amount(self, engineer):
        """Test handling of zero amounts"""
        data = pd.DataFrame({
            'Time': [0],
            'Amount': [0.0]
        })
        for i in range(1, 29):
            data[f'V{i}'] = 0
        
        result = engineer.generate_all_features(data, is_batch=False)
        
        # Should handle without errors
        assert result is not None
        assert result['amount_log'].iloc[0] == np.log1p(0)  # log(1+0) = 0
    
    def test_large_amount(self, engineer):
        """Test handling of very large amounts"""
        data = pd.DataFrame({
            'Time': [0],
            'Amount': [1e6]  # 1 million
        })
        for i in range(1, 29):
            data[f'V{i}'] = 0
        
        result = engineer.generate_all_features(data, is_batch=False)
        
        assert result is not None
        assert result['amount_is_huge'].iloc[0] == 1
    
    def test_empty_dataframe(self, engineer):
        """Test handling of empty DataFrame"""
        data = pd.DataFrame(columns=['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)])
        
        result = engineer.generate_all_features(data, is_batch=True)
        
        # Should return empty DataFrame with all columns
        assert result is not None
        assert len(result) == 0
        assert len(result.columns) > 30


# ==================== PERFORMANCE TESTS ====================

def test_performance_batch(engineer):
    """Test batch processing performance"""
    import time
    
    # Create large dataset
    np.random.seed(42)
    n = 10000
    data = {
        'Time': np.sort(np.random.randint(0, 172800, n)),
        'Amount': np.random.exponential(100, n)
    }
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n)
    
    df = pd.DataFrame(data)
    
    # Time the operation
    start = time.time()
    result = engineer.generate_all_features(df, is_batch=True)
    elapsed = time.time() - start
    
    # Should process 10K transactions in reasonable time
    assert elapsed < 30  # Less than 30 seconds
    assert len(result) == n
    
    print(f"\n⏱️  Processed {n:,} transactions in {elapsed:.2f}s ({n/elapsed:.0f} txn/sec)")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
