import pandas as pd
import numpy as np

print("Checking dataset...")
df = pd.read_csv('data/creditcard.csv')

print(f"\n✅ Dataset loaded!")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Fraud count: {df['Class'].sum():,}")
print(f"Fraud rate: {df['Class'].mean()*100:.3f}%")

print(f"\nColumns: {df.columns.tolist()}")

# Check for any data issues
print(f"\nNull values: {df.isnull().sum().sum()}")
print(f"Data types unique: {df.dtypes.unique()}")
