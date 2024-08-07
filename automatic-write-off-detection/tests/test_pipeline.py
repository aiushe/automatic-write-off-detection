import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from pipeline import preprocess_data, train_model


def test_pipeline():
    #sample data
    data = {
        'plaid_merchant_description': [
            'TRELLO.COM* ATLASSIAN XXXXXXXXXXXX0497',
            'MOBILE PURCHASE 01/03 MARATHON PETRO11',
            'PGANDE DES:WEB ONLINE ID:XXXXXXXXXX8324 INDN:JOHNNY CO ID:XXXXX11632 WEB',
            'PMNT SENT 0104 APPLE CASH - SENT CA XXXXX3630XXXXXXXXXX8324',
            '3683 MJ4LBR3A AMAZON.COMH51UG38I0 AMAZON.COM SEATTLEWA C# 0497'
        ],
        'keeper_category': [
            '💻 software',
            '⛽ gas fill up',
            '🏠 utilities',
            '↔️ transfer',
            '📦 supplies'
        ]
    }
    df = pd.DataFrame(data)

    #initialize vectorizer
    vectorizer = TfidfVectorizer()

    #preprocess data
    X, y = preprocess_data(df, vectorizer)

    #train
    model = train_model(X, y, model_type='naive_bayes')

    #test
    y_pred = model.predict(X)

    print(classification_report(y, y_pred, zero_division=1))


if __name__ == "__main__":
    test_pipeline()