import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import openai
import re
from fuzzywuzzy import fuzz, process
import joblib

openai.api_key = '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'

def extract_entities(transaction):
    # disect transaction data following the pattern (MetaData, Bank, App) and ignore the rest
    pattern = {
        'metadata': r'(DEBIT WITHDRAWAL|PURCHASE|PRE-AUTHORIZATION DEBIT|POS DEBIT)',
        'bank': r'(WELLS FARGO|BANK OF AMERICA|CHASE|CITI|BP)',
        'app': r'(UBER|LYFT|PAYPAL|AFFIRM|APPLE)'
    }

    entities = {'metadata': None, 'bank': None, 'app': None}

    # loop throught the string to find and determine the important info defined in the pattern above
    for key, p in pattern.items():
        match = re.search(p, transaction, re.IGNORECASE)
        if match:
            entities[key] = match.group(0)
    return entities

def prioritize(entities):
    if entities['app']:
        return entities['app']
    elif entities['bank']:
        return entities['bank']
    else:
        if entities['metadata']:
            return entities['metadata']
        else:
            'UNKNOWN'

def identify_merchant(transaction, sample):
    #use fuzz to determine score threshold(if the best match is found, but it is not greater than this number, then return None otherwise default 0)
    match = process.extractOne(transaction, sample, scorer=fuzz.token_sort_ratio)
    if match:
        return match[0]
    else:
        return 'UNKNOWN'

def gpt_feature_extraction(transaction):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract meaningful features from this transaction description."},
            {"role": "user", "content": transaction}
        ]
    )
    return response.choices[0].message['content'].strip()

def preprocess_data(df, vectorizer):
    df['merchant'] = df['transaction_info'].apply(lambda x: prioritize(extract_entities(x)))
    df['gpt_features'] = df['transaction_info'].apply(gpt_feature_extraction)
    X = vectorizer.fit_transform(df['gpt_features'])

    #categories
    unique_categories = df['category'].unique()
    print(f"Unique Categories: {unique_categories}")

    y = df['category']
    return X, y


def train_model(X, y, model_type='naive_bayes'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'linear_regression':
        model = LogisticRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model


def main():
    df = pd.read_csv('../data/expanded_transactions.csv')
    vectorizer = TfidfVectorizer()

    X, y = preprocess_data(df, vectorizer)
    model = train_model(X, y, model_type='naive_bayes')

    joblib.dump(model, '../models/transaction_classifier.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')


if __name__ == "__main__":
    main()