import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
from fuzzywuzzy import fuzz, process
from transformers import pipeline as transformers_pipeline, AutoTokenizer, AutoModelForSequenceClassification
import joblib

def identify_merchant(transaction):
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
        'UNKNOWN'

def categorize_transaction(transaction, vectorizer, model):
    new_info_data = vectorizer.transform([transaction])
    return model.predict(new_info_data)[0]

def find_duplicate(transaction1, transaction2, vectorizer, model):
    new_info_data = vectorizer.transform([transaction1, transaction2])
    return model.predict(new_info_data)

#implement transformer for ChatGPT4

def gpt_feature_extraction(transaction, nlp_pipline):
    features = nlp_pipline(transaction)
    return features[0][0]

def preprocess_data(df, vectorizer, nlp_pipline):
    df['merchant'] = df['transaction_info'].apply(lambda x: prioritize(extract_entities(x)))
    df['gpt_features'] = df['transaction_info'].apply(lambda x: gpt_feature_extraction(x, nlp_pipeline))
    X = list(df['gpt_features'])
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
    df = pd.read_csv('../data/transactions.csv')
    vectorizer = TfidfVectorizer()

    X, y = preprocess_data(df, vectorizer)
    model = train_model(X, y, model_type='naive_bayes')

    joblib.dump(model, '../models/transaction_classifier.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')


if __name__ == "__main__":
    main()