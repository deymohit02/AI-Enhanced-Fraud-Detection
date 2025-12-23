"""
Optimized Training Script for Fraud Detection Model
- Hyperparameter tuning with RandomizedSearchCV
- Multiple model comparison
- Stratified cross-validation
- Comprehensive metrics and visualization
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import sys

# Import internal modules
sys.path.append('.')
from BalanceDataset import balanceWithSMOTE

# Try to import optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️  Matplotlib/Seaborn not found. Plots will be skipped.")
    PLOTTING_AVAILABLE = False

# Constants
DATA_PATH = 'data/creditcard.csv'
MODELS_DIR = 'models'
RESULTS_DIR = 'training_results'

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Load the credit card fraud dataset"""
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERROR: Dataset not found at {DATA_PATH}")
        print("\nPlease download the dataset first:")
        print("1. Run: python download_kaggle_dataset.py")
        print("2. OR manually download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"3. Place creditcard.csv in: {os.path.abspath('data/')}")
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

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance training data"""
    print("\n⚖️  Applying SMOTE for class balancing...")
    
    fraud_before = y_train.sum()
    normal_before = (y_train == 0).sum()
    print(f"   Before SMOTE - Fraud: {fraud_before:,}, Normal: {normal_before:,}")
    
    X_balanced, y_balanced = balanceWithSMOTE(X_train, y_train)
    
    fraud_after = y_balanced.sum()
    normal_after = (y_balanced == 0).sum()
    print(f"   After SMOTE - Fraud: {fraud_after:,}, Normal: {normal_after:,}")
    print(f"   ✅ Data balanced!")
    
    return X_balanced, y_balanced

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\n📈 Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),  # FIX: Use probabilities, not predictions
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Print results
    print(f"\n   Results for {model_name}:")
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
    
    return metrics, y_pred, y_pred_proba

def plot_results(y_test, y_pred_proba, model_name, metrics):
    """Plot ROC and Precision-Recall curves"""
    if not PLOTTING_AVAILABLE:
        return
    
    print(f"\n📊 Generating plots for {model_name}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}", linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1].plot(recall, precision, linewidth=2)
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, f'{model_name.lower().replace(" ", "_")}_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Plots saved to: {plot_path}")
    plt.close()

def train_xgboost_optimized(X_train, X_test, y_train, y_test):
    """Train XGBoost with hyperparameter tuning"""
    print("\n" + "="*70)
    print("🚀 TRAINING XGBOOST WITH HYPERPARAMETER TUNING")
    print("="*70)
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / y_train.sum()
    print(f"\nClass imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # Hyperparameter grid
    param_distributions = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    # Base model
    xgb_base = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    # Randomized search with stratified k-fold
    print("\n⏳ Running RandomizedSearchCV (this may take 10-20 minutes)...")
    print("   - 20 parameter combinations")
    print("   - 3-fold stratified cross-validation")
    print("   - Optimization metric: ROC AUC\n")
    
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        xgb_base,
        param_distributions,
        n_iter=20,
        scoring='roc_auc',
        cv=stratified_kfold,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Hyperparameter tuning complete!")
    print(f"\n   Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"      {param}: {value}")
    print(f"\n   Best CV AUC Score: {random_search.best_score_:.4f}")
    
    # Best model
    best_model = random_search.best_estimator_
    
    # Evaluate
    metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, "XGBoost Optimized")
    
    # Add hyperparameters to metrics
    metrics['best_params'] = random_search.best_params_
    metrics['cv_auc'] = random_search.best_score_
    
    # Plot
    plot_results(y_test, y_pred_proba, "XGBoost Optimized", metrics)
    
    return best_model, metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest as comparison"""
    print("\n" + "="*70)
    print("🌲 TRAINING RANDOM FOREST (for comparison)")
    print("="*70)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n⏳ Training Random Forest...")
    rf.fit(X_train, y_train)
    
    metrics, y_pred, y_pred_proba = evaluate_model(rf, X_test, y_test, "Random Forest")
    plot_results(y_test, y_pred_proba, "Random Forest", metrics)
    
    return rf, metrics

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
        elif isinstance(v, dict):
            serializable_metrics[k] = {
                str(k2): float(v2) if isinstance(v2, (np.float32, np.float64)) else v2
                for k2, v2 in v.items()
            }
        else:
            serializable_metrics[k] = v
    
    serializable_metrics['training_date'] = datetime.now().isoformat()
    
    with open(metrics_path, 'w', indent=2) as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"📊 Metrics saved to: {metrics_path}")

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🤖 FRAUD DETECTION MODEL - OPTIMIZED TRAINING PIPELINE")
    print("="*70)
    print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    # Train XGBoost with hyperparameter tuning
    xgb_model, xgb_metrics = train_xgboost_optimized(
        X_train_balanced, X_test, y_train_balanced, y_test
    )
    save_model(xgb_model, "xgboost", xgb_metrics)
    
    # Train Random Forest for comparison
    rf_model, rf_metrics = train_random_forest(
        X_train_balanced, X_test, y_train_balanced, y_test
    )
    save_model(rf_model, "random_forest", rf_metrics)
    
    # Compare models
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    
    comparison = pd.DataFrame([
        {
            'Model': 'XGBoost Optimized',
            'AUC': xgb_metrics['auc'],
            'Precision': xgb_metrics['precision'],
            'Recall': xgb_metrics['recall'],
            'F1-Score': xgb_metrics['f1']
        },
        {
            'Model': 'Random Forest',
            'AUC': rf_metrics['auc'],
            'Precision': rf_metrics['precision'],
            'Recall': rf_metrics['recall'],
            'F1-Score': rf_metrics['f1']
        }
    ])
    
    print("\n" + comparison.to_string(index=False))
    
    # Determine best model
    best_model_name = 'XGBoost' if xgb_metrics['auc'] > rf_metrics['auc'] else 'Random Forest'
    best_auc = max(xgb_metrics['auc'], rf_metrics['auc'])
    
    print(f"\n🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")
    
    # Check if target achieved
    print("\n" + "="*70)
    print("🎯 TARGET METRICS EVALUATION")
    print("="*70)
    
    targets = {
        'AUC > 0.90': best_auc >= 0.90,
        'Precision > 0.80': max(xgb_metrics['precision'], rf_metrics['precision']) >= 0.80,
        'Recall > 0.75': max(xgb_metrics['recall'], rf_metrics['recall']) >= 0.75,
        'F1 > 0.77': max(xgb_metrics['f1'], rf_metrics['f1']) >= 0.77
    }
    
    for target, achieved in targets.items():
        status = "✅ ACHIEVED" if achieved else "❌ NOT MET"
        print(f"   {target:<20} {status}")
    
    if all(targets.values()):
        print("\n🎉 SUCCESS! All target metrics achieved!")
    else:
        print("\n⚠️  Some targets not met. Consider:")
        print("   - More hyperparameter tuning iterations")
        print("   - Trying different balancing techniques")
        print("   - Feature engineering")
    
    print(f"\n✅ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70)
    print("📁 NEXT STEPS")
    print("="*70)
    print("\n1. Review metrics in: models/xgboost_metrics.json")
    print("2. Check plots in: training_results/")
    print("3. Test the API: python test_api_with_real_data.py")
    print("4. Deploy the updated model")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
