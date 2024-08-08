import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
from merchant_identification import load_merchant_entities, find_best_match


#load models
vectorizer = joblib.load('../models/vectorizer.pkl')
model = joblib.load('../models/transaction_classifier.pkl')
merchant_df = load_merchant_entities('../data/merchant_entities.csv')

def categorize_transaction(transactions):
    merchant, category = find_best_match(transactions, merchant_df)

    if category != 'UNKNOWN':
        return category
    else:
        new_info_data = vectorizer.transform([transactions])
        return model.predict(new_info_data)[0]

if __name__ == "__main__":
    new_transactions = [
        'DEBIT CARD DEBIT / AUTH #710620 03-01-2023 BP#5364328RPF',
        'BP PRODUCTS NORTH AMERICA'
    ]
    for t in new_transactions:
        category = categorize_transaction(t)
        print(f"Transaction: {t}")
        print(f"Category: {category}")
        print("-" * 40)