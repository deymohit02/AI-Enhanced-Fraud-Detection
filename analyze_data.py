import pandas as pd

df = pd.read_csv('data/creditcard.csv')
print(f'Total rows: {len(df)}')
print(f'Fraud cases: {df["Class"].sum()} ({df["Class"].sum()/len(df)*100:.2f}%)')
print(f'Non-fraud: {(df["Class"]==0).sum()}')
print('\nColumns:')
print(df.columns.tolist())
print('\nSample non-fraud transaction:')
print(df[df['Class']==0].iloc[0])
print('\nSample fraud transaction:')
if df['Class'].sum() > 0:
    print(df[df['Class']==1].iloc[0])
print('\nAmount statistics:')
print(df['Amount'].describe())
