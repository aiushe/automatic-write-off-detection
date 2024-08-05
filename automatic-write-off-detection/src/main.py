from merchant_identification import extract_entities, prioritize, identify_merchant
from categorization import categorize_transaction
from duplicate_handling import find_duplicate
import pandas as pd

#load data
df = pd.read_csv('../data/transactions.csv')

#identify merchants
df['entities'] = df['transaction_info'].apply(extract_entities)
df['merchant'] = df.apply(lambda x: prioritize(x['entities']), axis = 1)

#categorize
df['category'] = df['transaction_info'].apply(categorize_transaction)

#find duplicates or unusual transactions
find_unusual_transactions = []

for t in df['transaction_info']:
    usual = True
    for ut in find_unusual_transactions:
        if find_duplicate(t, ut):
            usual = False
            break
        if usual:
            find_unusual_transactions.append(t)

print(df[['transaction_info', 'merchant', 'category']])
print("unique transactions:")
print(find_unusual_transactions)