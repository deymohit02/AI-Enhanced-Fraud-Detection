import pandas as pd
import numpy as np

df = pd.read_csv('data/creditcard.csv')

print("=== DATA ANALYSIS ===")
print(f"\nDataset Shape: {df.shape}")
print(f"Fraud Rate: {df['Class'].sum()}/{len(df)} = {df['Class'].mean()*100:.2f}%")

print("\n=== V-FEATURES STATISTICS ===")
print("\nNon-Fraud V-features mean:")
v_cols = [f'V{i}' for i in range(1, 29)]
non_fraud = df[df['Class']==0]
fraud = df[df['Class']==1]

for col in v_cols[:5]:  # First 5 V features
    print(f"{col}: Non-fraud mean={non_fraud[col].mean():.4f}, Fraud mean={fraud[col].mean():.4f}")

print("\n=== CHECKING IF V-FEATURES ARE ALL ZERO ===")
# Check how many transactions have all V-features as zero
all_v_zero = (df[v_cols] == 0).all(axis=1)
print(f"Transactions with all V-features = 0: {all_v_zero.sum()} ({all_v_zero.sum()/len(df)*100:.2f}%)")

if all_v_zero.sum() > 0:
    print(f"Of these, frauds: {df[all_v_zero]['Class'].sum()}")
    print(f"Of these, non-frauds: {(df[all_v_zero]['Class']==0).sum()}")

print("\n=== V-FEATURES RANGE ===")
print("\nNon-zero V-features analysis:")
for col in v_cols[:5]:
    print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, std={df[col].std():.2f}")
