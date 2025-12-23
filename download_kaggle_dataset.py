"""
Automated Kaggle Dataset Download Script
Downloads the Credit Card Fraud Detection dataset from Kaggle
"""

import os
import sys
import zipfile
import subprocess

def check_kaggle_installed():
    """Check if Kaggle package is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        return False

def install_kaggle():
    """Install Kaggle package"""
    print("📦 Installing Kaggle package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    print("✅ Kaggle package installed successfully!")

def check_kaggle_credentials():
    """Check if Kaggle API credentials exist"""
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        return True
    return False

def setup_instructions():
    """Print setup instructions for Kaggle API"""
    print("\n" + "="*70)
    print("📋 KAGGLE API SETUP INSTRUCTIONS")
    print("="*70)
    print("\n1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. This will download 'kaggle.json'")
    print("\n5. Move 'kaggle.json' to:")
    print(f"   {os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')}")
    print("\n6. Create the directory if it doesn't exist:")
    print(f"   mkdir {os.path.join(os.path.expanduser('~'), '.kaggle')}")
    print("\n" + "="*70)
    print("\nAfter setup, run this script again!")
    print("="*70 + "\n")

def download_dataset():
    """Download the Credit Card Fraud dataset from Kaggle"""
    import kaggle
    
    dataset_name = "mlg-ulb/creditcardfraud"
    download_path = "data"
    
    print("\n" + "="*70)
    print("📥 DOWNLOADING KAGGLE DATASET")
    print("="*70)
    print(f"\nDataset: {dataset_name}")
    print(f"Destination: {download_path}/\n")
    
    # Create data directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Download dataset
    print("⏳ Downloading... (this may take a few minutes)")
    kaggle.api.dataset_download_files(
        dataset_name,
        path=download_path,
        unzip=True
    )
    
    print("\n✅ Download complete!")
    return os.path.join(download_path, "creditcard.csv")

def verify_dataset(csv_path):
    """Verify the downloaded dataset"""
    import pandas as pd
    
    print("\n" + "="*70)
    print("🔍 VERIFYING DATASET")
    print("="*70)
    
    if not os.path.exists(csv_path):
        print(f"\n❌ File not found: {csv_path}")
        return False
    
    # Load and analyze dataset
    print(f"\n📊 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"\nShape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    # Check for expected columns
    expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    missing_cols = set(expected_cols) - set(df.columns)
    
    if missing_cols:
        print(f"\n⚠️  Missing columns: {missing_cols}")
        return False
    
    # Display fraud statistics
    fraud_count = df['Class'].sum()
    normal_count = (df['Class'] == 0).sum()
    fraud_rate = fraud_count / len(df) * 100
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Normal transactions: {normal_count:,} ({100-fraud_rate:.2f}%)")
    print(f"   Fraudulent transactions: {fraud_count:,} ({fraud_rate:.2f}%)")
    
    print(f"\n💰 Amount Statistics:")
    print(f"   Min: ${df['Amount'].min():.2f}")
    print(f"   Max: ${df['Amount'].max():.2f}")
    print(f"   Mean: ${df['Amount'].mean():.2f}")
    print(f"   Median: ${df['Amount'].median():.2f}")
    
    # File size
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"\n📦 File size: {file_size_mb:.2f} MB")
    
    # Expected dataset characteristics
    expected_rows = 284807
    if len(df) != expected_rows:
        print(f"\n⚠️  Warning: Expected {expected_rows:,} rows, got {len(df):,}")
    else:
        print(f"\n✅ Row count matches expected: {expected_rows:,}")
    
    if abs(fraud_rate - 0.172) > 0.01:
        print(f"⚠️  Warning: Expected fraud rate ~0.172%, got {fraud_rate:.3f}%")
    else:
        print(f"✅ Fraud rate matches expected: ~0.172%")
    
    print("\n" + "="*70)
    print("✅ DATASET VERIFICATION COMPLETE")
    print("="*70 + "\n")
    
    return True

def main():
    """Main function to orchestrate dataset download"""
    print("\n" + "="*70)
    print("🤖 KAGGLE CREDIT CARD FRAUD DATASET DOWNLOADER")
    print("="*70 + "\n")
    
    # Step 1: Check if kaggle is installed
    if not check_kaggle_installed():
        print("❌ Kaggle package not found")
        response = input("Would you like to install it now? (y/n): ").strip().lower()
        if response == 'y':
            install_kaggle()
        else:
            print("\n⚠️  Please install kaggle package: pip install kaggle")
            return
    else:
        print("✅ Kaggle package is installed")
    
    # Step 2: Check for credentials
    if not check_kaggle_credentials():
        print("\n❌ Kaggle API credentials not found")
        setup_instructions()
        return
    else:
        print("✅ Kaggle API credentials found")
    
    # Step 3: Check if dataset already exists
    csv_path = "data/creditcard.csv"
    if os.path.exists(csv_path):
        print(f"\n⚠️  Dataset already exists at: {csv_path}")
        response = input("Would you like to re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("\n✅ Using existing dataset")
            verify_dataset(csv_path)
            return
    
    # Step 4: Download dataset
    try:
        csv_path = download_dataset()
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Kaggle API credentials are correct")
        print("3. Make sure you've accepted the dataset terms on Kaggle website:")
        print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return
    
    # Step 5: Verify dataset
    if verify_dataset(csv_path):
        print("\n🎉 SUCCESS! Dataset is ready for model training.")
        print(f"\nNext steps:")
        print("1. Run: python train_optimized.py")
        print("2. Wait for training to complete (~5-10 minutes)")
        print("3. Check models/xgboost_metrics.json for performance")
    else:
        print("\n❌ Dataset verification failed")

if __name__ == "__main__":
    main()
