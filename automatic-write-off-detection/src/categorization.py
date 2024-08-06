import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

vectorizer = joblib.load('../models/vectorizer.pkl')
model = joblib.load('../models/transaction_classifier.pkl')

def categorize_transaction(transactions):
    new_info_data = vectorizer.transform([transactions])
    return model.predict(new_info_data)[0]

#test
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
'''
Transaction: DEBIT CARD DEBIT / AUTH #710620 03-01-2023 BP#5364328RPF
Category: Debit
----------------------------------------
Transaction: BP PRODUCTS NORTH AMERICA
Category: Purchase
----------------------------------------
'''