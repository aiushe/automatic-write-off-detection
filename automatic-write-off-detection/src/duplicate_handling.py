import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#load models
vectorizer = joblib.load('../models/vectorizer_expanded.pkl')
model = joblib.load('../models/transaction_classifier_expanded.pkl')


def find_duplicate(transaction1, transaction2):
    new_info_data = vectorizer.transform([transaction1, transaction2])
    pre = model.predict(new_info_data)
    return pre[0] == pre[1]


if __name__ == "__main__":
    new_transactions = [
        'PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX',
        'BP PRODUCTS NORTH AMERICA'
    ]
    duplicates = find_duplicate(*new_transactions)
    print(f"Transaction: {new_transactions}")
    if duplicates[0] == 1:
        print("Yes")
    else:
        print("No")

