import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
try:
    import tensorflow as tf
    from DeepLearningModels import ANN_model, CNN_model
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not found. DL models will be skipped.")
    TF_AVAILABLE = False

# Import internal modules
import sys
sys.path.append('.')
from BalanceDataset import balanceWithSMOTE
from MachineLearningModels import model_performance

# Constants
DATA_PATH = 'data/creditcard.csv'
MODELS_DIR = 'models'

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic credit card data if file not found"""
    print("Dataset not found! Generating synthetic data for demonstration...")
    
    # 28 V-features, Time, Amount, Class
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    
    df = pd.DataFrame(np.random.randn(n_samples, 30), columns=columns[:-1])
    # Add Time (sequential)
    df['Time'] = range(n_samples)
    # Add Amount (positive)
    df['Amount'] = df['Amount'].abs() * 100
    # Add Class (mostly 0, some 1)
    df['Class'] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Save to data directory for future use
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic data saved to {DATA_PATH}")
    return df

def load_data():
    if not os.path.exists(DATA_PATH):
        # Look in original structure
        if os.path.exists('creditcard/creditcard.csv'):
            return pd.read_csv('creditcard/creditcard.csv')
        else:
            return generate_synthetic_data()
    return pd.read_csv(DATA_PATH)

def save_model(model, name, metrics, is_dl=False):
    """Save model and metrics"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    
    if is_dl:
        # Save Keras model
        model_path = os.path.join(MODELS_DIR, f"{name}.h5")
        model.save(model_path)
        print(f"Saved DL model to {model_path}")
    else:
        # Save sklearn/xgboost model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved ML model to {path}")
        
    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, f"{name}_metrics.json")
    
    # Convert numpy types to native python types for JSON serialization
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64)):
            serializable_metrics[k] = float(v)
        else:
            serializable_metrics[k] = v
            
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f)
    print(f"Saved metrics to {metrics_path}")

def main():
    print("1. Loading Data...")
    df = load_data()
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print("2. Preprocessing...")
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for inference
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler to models/scaler.pkl")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("3. Balancing Data (SMOTE)...")
    # Note: balanceWithSMOTE returns balanced X and y
    X_train_bal, y_train_bal = balanceWithSMOTE(X_train, y_train)
    
    print("\n--- Training Models ---")
    
    # 1. XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss')
    xgb_model, xgb_metrics = model_performance(xgb, X_train_bal, X_test, y_train_bal, y_test, "SMOTE")
    save_model(xgb_model, "xgboost", xgb_metrics, is_dl=False)
    
    if TF_AVAILABLE:
        # 2. ANN
        print("\nTraining ANN...")
        ann_model, ann_metrics = ANN_model(X_train_bal, X_test, y_train_bal, y_test)
        # The returned 'metrics' is actually a tuple: ("ANN", precision, recall, f1, auc)
        # We need to convert it to dict
        ann_metrics_dict = {
            "precision": ann_metrics[1],
            "recall": ann_metrics[2],
            "f1": ann_metrics[3],
            "auc": ann_metrics[4]
        }
        save_model(ann_model, "ann", ann_metrics_dict, is_dl=True)
        
        # 3. CNN
        print("\nTraining CNN...")
        cnn_model, cnn_metrics = CNN_model(X_train_bal, X_test, y_train_bal, y_test)
        cnn_metrics_dict = {
            "precision": cnn_metrics[1],
            "recall": cnn_metrics[2],
            "f1": cnn_metrics[3],
            "auc": cnn_metrics[4]
        }
        save_model(cnn_model, "cnn", cnn_metrics_dict, is_dl=True)
    else:
        print("\nSkipping DL models (TensorFlow missing).")
    
    print("\n✅ Training Complete. Models saved in 'models/' directory.")

if __name__ == "__main__":
    main()
