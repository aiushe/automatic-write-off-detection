import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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