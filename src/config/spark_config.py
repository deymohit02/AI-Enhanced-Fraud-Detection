"""
Apache Spark Configuration Module
Manages Spark session initialization for distributed fraud detection processing
"""

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import os

class SparkConfig:
    """Spark session manager for fraud detection"""
    
    _instance = None
    _spark = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SparkConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._spark is None:
            self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session with optimized configuration"""
        
        conf = SparkConf()
        
        # Application settings
        conf.set("spark.app.name", "FraudDetectionPipeline")
        
        # Executor configuration (adjust based on your system)
        conf.set("spark.executor.memory", "4g")
        conf.set("spark.driver.memory", "4g")
        conf.set("spark.executor.cores", "4")
        
        # Optimize for local development (change for production cluster)
        conf.set("spark.master", "local[*]")  # Use all available cores
        
        # Performance tuning
        conf.set("spark.sql.shuffle.partitions", "200")
        conf.set("spark.default.parallelism", "8")
        
        # Memory management
        conf.set("spark.memory.fraction", "0.8")
        conf.set("spark.memory.storageFraction", "0.3")
        
        # Serialization (Kryo is faster than Java serialization)
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        conf.set("spark.kryo.registrationRequired", "false")
        
        # Checkpoint directory (for streaming)
        checkpoint_dir = os.path.join(os.getcwd(), "processed_data", "spark_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        conf.set("spark.sql.streaming.checkpointLocation", checkpoint_dir)
        
        # UI configuration
        conf.set("spark.ui.enabled", "true")
        conf.set("spark.ui.port", "4040")
        
        # Create Spark session
        self._spark = SparkSession.builder \
            .config(conf=conf) \
            .getOrCreate()
        
        # Configure logging
        self._spark.sparkContext.setLogLevel("WARN")  # Reduce log verbosity
        
        print(f"✅ Spark session initialized")
        print(f"   Master: {self._spark.sparkContext.master}")
        print(f"   App Name: {self._spark.sparkContext.appName}")
        print(f"   Spark UI: http://localhost:4040")
    
    def get_spark_session(self):
        """Get the Spark session instance"""
        return self._spark
    
    def stop(self):
        """Stop the Spark session"""
        if self._spark:
            self._spark.stop()
            print("🛑 Spark session stopped")
            self._spark = None

# Singleton instance
spark_config = SparkConfig()

def get_spark():
    """Convenience function to get Spark session"""
    return spark_config.get_spark_session()
