import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
            {"role": "system", "content": "xxtract features from this transaction description."},
            {"role": "user", "content": transaction}
        ]
    )
    return response['choices'][0]['message']['content']

def preprocess_data(df, vectorizer):
    df['gpt_features'] = df['plaid_merchant_description'].apply(gpt_feature_extraction)
    df['merchant'] = df['plaid_merchant_description'].apply(lambda x: prioritize(extract_entities(x)))
    X = vectorizer.fit_transform(df['plaid_merchant_description'] + ' ' + df['gpt_features'] + ' ' + df['merchant'])

    #categories
    #unique_categories = df['category'].unique()
    #print(f"Unique Categories: {unique_categories}")

    y = df['keeper_category'].fillna('')
    return X, y

def train_model(X, y, model_type='naive_bayes'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'linear_regression':
        model = LogisticRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #classification report
    print(classification_report(y_test, y_pred, zero_division=1))
    print(confusion_matrix(y_test, y_pred))

    return model

def main():
    df = pd.read_csv('../data/expanded_transactions.csv')
    print("columns:", df.columns)
    vectorizer = TfidfVectorizer()
    X, y = preprocess_data(df, vectorizer)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    #classification_report
    print(classification_report(y_test, y_pred))

    joblib.dump(model, '../models/transaction_classifier.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')


if __name__ == "__main__":
    main()