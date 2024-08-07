from merchant_identification import extract_entities, prioritize, identify_merchant
from categorization import categorize_transaction
from duplicate_handling import find_duplicate
import pandas as pd
import joblib
import openai
from utils import load_env_vars
from logger import logger

openai_api_key = load_env_vars()
openai.api_key = openai_api_key


#load vectorizer and model
vectorizer = joblib.load('../models/vectorizer.pkl')
model = joblib.load('../models/transaction_classifier.pkl')

#load data
df = pd.read_csv('../data/expanded_transactions.csv')

#identify merchants
df['entities'] = df['plaid_merchant_description'].apply(extract_entities)
df['merchant'] = df.apply(lambda x: prioritize(x['entities']), axis = 1)

#categorize
df['category'] = df['plaid_merchant_description'].apply(categorize_transaction)

#find duplicates or unusual transactions
find_unusual_transactions = []

for t in df['plaid_merchant_description']:
    usual = True
    for ut in find_unusual_transactions:
        if find_duplicate(t, ut):
            usual = False
            break
        if usual:
            find_unusual_transactions.append(t)

logger.info("Processed transactions:")
logger.info(df[['plaid_merchant_description', 'merchant', 'category']])
logger.info("Unique transactions:")
logger.info(find_unusual_transactions)