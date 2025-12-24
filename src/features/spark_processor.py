"""
Apache Spark Feature Processor
Distributed feature engineering for batch and real-time processing
"""

import os
import sys
from typing import Dict, Optional, List
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config.spark_config import get_spark
from src.features.feature_engineering import FeatureEngineer


class SparkFeatureProcessor:
    """
    Spark-based feature engineering processor for large-scale batch processing
    and real-time feature computation.
    """
    
    def __init__(self):
        """Initialize Spark feature processor"""
        self.spark = None
        self.feature_engineer = FeatureEngineer(cache_enabled=True)
        self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session"""
        try:
            self.spark = get_spark()
            print("✅ Spark Feature Processor initialized")
        except Exception as e:
            print(f"⚠️  Spark initialization failed: {e}")
            print("   Falling back to pandas-only processing")
            self.spark = None
    
    def process_batch(self, input_path: str, output_path: str, 
                     file_format: str = 'csv') -> bool:
        """
        Process a batch of transactions with feature engineering.
        
        Args:
            input_path: Path to input data (CSV or Parquet)
            output_path: Path to save processed data
            file_format: Output format ('csv' or 'parquet')
            
        Returns:
            True if successful, False otherwise
        """
        if self.spark is None:
            print("❌ Spark not available, using pandas fallback")
            return self._process_batch_pandas(input_path, output_path, file_format)
        
        try:
            print(f"\n{'='*70}")
            print("🚀 SPARK BATCH FEATURE PROCESSING")
            print(f"{'='*70}")
            print(f"Input:  {input_path}")
            print(f"Output: {output_path}")
            print(f"Format: {file_format}")
            
            # Load data
            print("\n📂 Loading data into Spark DataFrame...")
            if input_path.endswith('.csv'):
                df_spark = self.spark.read.csv(input_path, header=True, inferSchema=True)
            elif input_path.endswith('.parquet'):
                df_spark = self.spark.read.parquet(input_path)
            else:
                raise ValueError(f"Unsupported file format: {input_path}")
            
            row_count = df_spark.count()
            print(f"   ✅ Loaded {row_count:,} transactions")
            
            # Validate schema
            print("\n🔍 Validating schema...")
            if not self._validate_schema(df_spark):
                raise ValueError("Invalid schema: missing required columns")
            print("   ✅ Schema validated")
            
            # Convert to pandas for feature engineering
            # (For very large datasets, implement Spark UDFs instead)
            print("\n⚙️  Converting to Pandas for feature engineering...")
            df_pandas = df_spark.toPandas()
            
            # Generate features
            print(f"\n🔧 Generating {self.feature_engineer.get_feature_count()} features...")
            df_features = self.feature_engineer.generate_all_features(df_pandas, is_batch=True)
            
            feature_count = len(df_features.columns)
            print(f"   ✅ Generated {feature_count} total columns")
            
            # Convert back to Spark and save
            print(f"\n💾 Saving processed data to {output_path}...")
            df_spark_out = self.spark.createDataFrame(df_features)
            
            if file_format == 'csv':
                df_spark_out.write.mode('overwrite').csv(output_path, header=True)
            elif file_format == 'parquet':
                df_spark_out.write.mode('overwrite').parquet(output_path)
            else:
                raise ValueError(f"Unsupported output format: {file_format}")
            
            print(f"   ✅ Data saved successfully")
            print(f"\n{'='*70}")
            print("✅ BATCH PROCESSING COMPLETE")
            print(f"{'='*70}\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_batch_pandas(self, input_path: str, output_path: str, 
                             file_format: str) -> bool:
        """Fallback batch processing using pandas only"""
        try:
            print(f"\n{'='*70}")
            print("🐼 PANDAS BATCH FEATURE PROCESSING (Fallback)")
            print(f"{'='*70}")
            
            # Load data
            print(f"\n📂 Loading {input_path}...")
            if input_path.endswith('.csv'):
                df = pd.read_csv(input_path)
            elif input_path.endswith('.parquet'):
                df = pd.read_parquet(input_path)
            else:
                raise ValueError(f"Unsupported format: {input_path}")
            
            print(f"   ✅ Loaded {len(df):,} transactions")
            
            # Generate features
            print(f"\n🔧 Generating features...")
            df_features = self.feature_engineer.generate_all_features(df, is_batch=True)
            print(f"   ✅ Generated {len(df_features.columns)} features")
            
            # Save
            print(f"\n💾 Saving to {output_path}...")
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                       exist_ok=True)
            
            if file_format == 'csv':
                df_features.to_csv(output_path, index=False)
            elif file_format == 'parquet':
                df_features.to_parquet(output_path, index=False)
            
            print(f"   ✅ Saved successfully")
            print(f"\n{'='*70}\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pandas processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compute_features_realtime(self, transaction: Dict) -> Optional[Dict]:
        """
        Compute features for a single transaction in real-time.
        
        Args:
            transaction: Dict with keys [Amount, Time, V1-V28]
            
        Returns:
            Dictionary with all features, or None if error
        """
        try:
            # Extract required fields
            amount = transaction.get('Amount', 0)
            time = transaction.get('Time', 0)
            v_features = [transaction.get(f'V{i}', 0) for i in range(1, 29)]
            
            # Use feature engineer to extract features
            features = self.feature_engineer.extract_features_for_api(
                amount=amount,
                time=time,
                v_features=v_features
            )
            
            return features
            
        except Exception as e:
            print(f"❌ Real-time feature computation failed: {e}")
            return None
    
    def _validate_schema(self, df: SparkDataFrame) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: Spark DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        df_columns = df.columns
        
        missing = [col for col in required_columns if col not in df_columns]
        
        if missing:
            print(f"   ❌ Missing columns: {missing}")
            return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get processor statistics"""
        return {
            'spark_available': self.spark is not None,
            'feature_count': self.feature_engineer.get_feature_count(),
            'cache_enabled': self.feature_engineer.cache_enabled
        }
    
    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            print("🛑 Spark session stopped")


# Convenience functions

def process_fraud_dataset(input_csv: str, output_csv: str) -> bool:
    """
    Process the fraud detection dataset with feature engineering.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        
    Returns:
        True if successful
    """
    processor = SparkFeatureProcessor()
    success = processor.process_batch(input_csv, output_csv, file_format='csv')
    processor.stop()
    return success


def compute_transaction_features(amount: float, time: float = 0,
                                 v_features: Optional[List[float]] = None) -> Optional[Dict]:
    """
    Compute features for a single transaction.
    
    Args:
        amount: Transaction amount
        time: Transaction time in seconds
        v_features: V1-V28 features (optional)
        
    Returns:
        Dictionary of computed features
    """
    processor = SparkFeatureProcessor()
    
    transaction = {
        'Amount': amount,
        'Time': time
    }
    
    if v_features:
        for i, v in enumerate(v_features, 1):
            transaction[f'V{i}'] = v
    else:
        for i in range(1, 29):
            transaction[f'V{i}'] = 0.0
    
    features = processor.compute_features_realtime(transaction)
    processor.stop()
    
    return features


if __name__ == "__main__":
    """CLI interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spark Feature Processing')
    parser.add_argument('--input', required=True, help='Input CSV/Parquet file')
    parser.add_argument('--output', required=True, help='Output CSV/Parquet file')
    parser.add_argument('--format', default='csv', choices=['csv', 'parquet'],
                       help='Output format')
    
    args = parser.parse_args()
    
    # Process
    processor = SparkFeatureProcessor()
    success = processor.process_batch(args.input, args.output, args.format)
    processor.stop()
    
    sys.exit(0 if success else 1)
