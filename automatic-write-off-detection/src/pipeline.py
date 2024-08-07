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

def extract_entities(transaction):
    pattern = {
        'transaction_type': r'(PURCHASE|DEBIT|POS|PRE-AUTHORIZATION|ATM WITHDRAWAL|TRANSFER)',
        'merchant': r'([A-Z][A-Za-z\s\d]*[A-Za-z])',  #mixed case and spaces/digits in merchant name
        'amount': r'[-]?\$\d+,\d+\.\d{2}',  #amounts like $1,960.00 or -$1,960.00
        'plaid_category': r"\['[^]]+'\]",  #category in brackets
    }

    entities = {'transaction_type': None, 'merchant': None, 'amount': None, 'plaid_category': None}

    #loop throught the string to find and determine the important info defined in the pattern above
    for key, p in pattern.items():
        match = re.search(p, transaction, re.IGNORECASE)
        if match:
            entities[key] = match.group(0)
    return entities


def prioritize(entities):
    if entities['merchant']:
        return entities['merchant']
    elif entities['transaction_type']:
        return entities['transaction_type']
    else:
            'UNKNOWN'


def identify_merchant(transaction, sample):
    # use fuzz to determine score threshold(if the best match is found, but it is not greater than this number,
    # then return None otherwise default 0)
    match = process.extractOne(transaction, sample, scorer=fuzz.token_sort_ratio)
    if match:
        return match[0]
    else:
        return 'UNKNOWN'


def gpt_feature_extraction(transaction):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "xxtract features from this transaction description."},
            {"role": "user", "content": transaction}
        ]
    )
    return response['choices'][0]['message']['content']


def preprocess_data(df, vectorizer):
    df['plaid_merchant_description'] = df['plaid_merchant_description'].fillna('')
    df['entities'] = df['plaid_merchant_description'].apply(lambda x: extract_entities(x))

    df['gpt_features'] = df['plaid_merchant_description'].apply(gpt_feature_extraction)

    df['merchant'] = df['entities'].apply(lambda x: prioritize(x)).fillna('')
    df['transaction_type'] = df['entities'].apply(lambda x: x['transaction_type']).fillna('')
    df['amount'] = df['entities'].apply(lambda x: x['amount']).fillna('')
    df['plaid_category'] = df['entities'].apply(lambda x: x['plaid_category']).fillna('')

    X = vectorizer.fit_transform(
        df['plaid_merchant_description'] + ' ' + df['gpt_features'] + ' ' + df['merchant'] + ' ' + df[
            'transaction_type'] + ' ' + df['amount'] + ' ' + df['plaid_category'])
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
    display_classification_report(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))

    return model


def display_classification_report(y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

    #convert to dataframe for easier readability
    report_df = pd.DataFrame(report_dict).transpose()

    print("\nclassification report:")
    print(report_df)


def main():
    df = pd.read_csv('../data/mini_transactions.csv')
    #print("columns:", df.columns)
    vectorizer = TfidfVectorizer()
    X, y = preprocess_data(df, vectorizer)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #classification_report
    display_classification_report(y_test, y_pred)
    '''
    model = train_model(X,y, model_type= 'naive_bayes')

    joblib.dump(model, '../models/transaction_classifier_mini.pkl')
    joblib.dump(vectorizer, '../models/vectorizer_mini.pkl')


if __name__ == "__main__":
    main()
