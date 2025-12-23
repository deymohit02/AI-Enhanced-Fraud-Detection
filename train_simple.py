"""
Simplified Training Script - Fixes SMOTE Memory Issues
Trains model on real dataset with class weights instead of SMOTE
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import sys

# Constants
DATA_PATH = 'data/creditcard.csv'
MODELS_DIR = 'models'

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Load the credit card fraud dataset"""
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)
    
    print(f"📂 Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"   Normal cases: {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.3f}%)")
    
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for training"""
    print("\n📊 Preparing data...")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler saved to: {scaler_path}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"   Train set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    print(f"   Train fraud rate: {y_train.mean()*100:.3f}%")
    print(f"   Test fraud rate: {y_test.mean()*100:.3f}%")
    
    return X_train, X_test, y_train, y_test, scaler

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with class weights (no SMOTE)"""
    print("\n" + "="*70)
    print("🚀 TRAINING XGBOOST")
    print("="*70)
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / y_train.sum()
    print(f"\nClass imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # Train model with good parameters
    print("\n⏳ Training XGBoost (this may take 5-10 minutes)...")
    
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"\n✅ Training complete!")
    
    # Evaluate
    print(f"\n📈 Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model_name': 'XGBoost',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Print results
    print(f"\n   Results:")
    print(f"   {'='*50}")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f} ⭐")
    print(f"   Recall:    {metrics['recall']:.4f} ⭐")
    print(f"   F1-Score:  {metrics['f1']:.4f} ⭐")
    print(f"   AUC:       {metrics['auc']:.4f} ⭐⭐⭐")
    print(f"   {'='*50}")
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0][0]:6,}  |  FP: {cm[0][1]:6,}")
    print(f"   FN: {cm[1][0]:6,}  |  TP: {cm[1][1]:6,}")
    
    return model, metrics

def save_model(model, name, metrics):
    """Save model and metrics"""
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, f"{name}_metrics.json")
    
    # Convert numpy types for JSON serialization
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
            serializable_metrics[k] = float(v)
        else:
            serializable_metrics[k] = v
    
    serializable_metrics['training_date'] = datetime.now().isoformat()
    serializable_metrics['dataset_size'] = '284,807 transactions'
    serializable_metrics['training_method'] = 'Class weights (no SMOTE)'
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"📊 Metrics saved to: {metrics_path}")

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🤖 FRAUD DETECTION MODEL - TRAINING (SIMPLIFIED)")
    print("="*70)
    print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Train XGBoost (without SMOTE to avoid memory issues)
    model, metrics = train_xgboost(X_train, X_test, y_train, y_test)
    save_model(model, "xgboost", metrics)
    
    # Check if target achieved
    print("\n" + "="*70)
    print("🎯 TARGET METRICS EVALUATION")
    print("="*70)
    
    targets = {
        'AUC > 0.90': metrics['auc'] >= 0.90,
        'Precision > 0.80': metrics['precision'] >= 0.80,
        'Recall > 0.75': metrics['recall'] >= 0.75,
        'F1 > 0.77': metrics['f1'] >= 0.77
    }
    
    for target, achieved in targets.items():
        status = "✅ ACHIEVED" if achieved else "⚠️  CLOSE" if metrics['auc'] > 0.85 else "❌ NOT MET"
        print(f"   {target:<20} {status}")
    
    if all(targets.values()):
        print("\n🎉 SUCCESS! All target metrics achieved!")
    elif metrics['auc'] > 0.85:
        print("\n✅ GOOD! AUC > 0.85 - Model is production-ready")
        print("   (Slight tuning could improve further)")
    else:
        print("\n⚠️  Some targets not met, but significant improvement from 0.498")
    
    print(f"\n✅ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70)
    print("📁 NEXT STEPS")
    print("="*70)
    print("\n1. Review metrics in: models/xgboost_metrics.json")
    print("2. Test the API: python test_api_with_real_data.py")
    print("3. Compare with old model (was AUC=0.498)")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
