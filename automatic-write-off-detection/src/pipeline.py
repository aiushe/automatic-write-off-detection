import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import openai
import re
from fuzzywuzzy import fuzz, process
import joblib

openai.api_key = 'sk-zJO5hTpYYWJqi9o-wacVM740H8EYFYHcfeHxU3r0VpT3BlbkFJE9AG2X5lhU2vMoyzDqAbqTF9Kx5dyrRyJPNpQc0QAA'

def extract(file_path):
    '''
    Extract data from a CSV file

    :param file_path: Path to the CSV file
    :return: DataFrame with extracted data
    '''
    return pd.read_csv(file_path)
def extract_entities(transaction):
    '''
    Extract entities from a transaction description using regular expressions

    :param transaction: Transaction description
    :return: Dictionary that contains the extracted entities
    '''
    pattern = {
        'transaction_type': r'(PURCHASE|DEBIT|POS|PRE-AUTHORIZATION|ATM WITHDRAWAL|TRANSFER)',
        'merchant': r'([A-Z][A-Za-z\s\d]*[A-Za-z])',  #mixed case and spaces/digits in merchant name
        'amount': r'[-]?\$\d+,\d+\.\d{2}',  #amounts like $1,960.00 or -$1,960.00
        'plaid_category': r"\['[^]]+'\]",  #category in brackets
    }

    entities = {'transaction_type': None, 'merchant': None, 'amount': None, 'plaid_category': None}

    #loop through pattern and extract entities from transaction description
    for key, p in pattern.items():
        match = re.search(p, transaction, re.IGNORECASE)
        if match:
            entities[key] = match.group(0)
    return entities


def prioritize(entities):
    '''
    Prioritize and return the most relevant information from the extracted entities

    :param entities: Dictionary that contains the extracted entities
    :return: Prioritized entity or 'UNKNOWN'
    '''
    if entities['merchant']:
        return entities['merchant']
    elif entities['transaction_type']:
        return entities['transaction_type']
    else:
        return 'UNKNOWN'


def identify_merchant(transaction, sample):
    '''
    Identify the merchant from transaction description using fuzzy matching

    :param transaction: Transaction description
    :param sample: List of sample merchants
    :return: Identified merchant or 'UNKNOWN'
    '''

    match = process.extractOne(transaction, sample, scorer=fuzz.token_sort_ratio)
    if match:
        return match[0]
    else:
        return 'UNKNOWN'


def gpt_feature_extraction(transaction):
    '''
    Extract features from a transaction description using OpenAI's GPT model

    :param transaction: Transaction description
    :return: Extracted features
    '''
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "xxtract features from this transaction description."},
            {"role": "user", "content": transaction}
        ]
    )
    return response['choices'][0]['message']['content']


def transform(df, vectorizer):
    '''
    Transform the data by extracting entities and generating features

    :param df: Input DataFrame.
    :param vectorizer: TF-IDF vectorizer
    :return: Tuple containing the feature matrix (X) and target vector (y)
    '''
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
    '''
    Train the model using navie bayes or linear regression

    :param X: Feature matrix
    :param y: Target vector
    :param model_type: Type of model to train ('naive_bayes' or 'linear_regression')
    :return: Trained model
    '''
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
    '''
    Display classification report in readable format

    :param y_test: True target values
    :param y_pred: Predicted target values.
    '''
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    report_df = pd.DataFrame(report_dict).transpose()

    print("\nclassification report:")
    print(report_df)


def load(model, vectorizer, model_path, vectorizer_path):
    '''
    Save trained model and vectorizer to disk

    :param model: Trained model
    :param vectorizer: TF-IDF vectorizer
    :param model_path: File path to save the model
    :param vectorizer_path: File path to save the vectorizer
    '''
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def main():
    '''
    Main function that extract data, transform, trains model, and save model
    '''
    #extract
    df = extract('../data/expanded_transactions.csv')

    #transform
    vectorizer = TfidfVectorizer()
    X, y = transform(df, vectorizer)

    #train model
    model = train_model(X, y, model_type='naive_bayes')

    #load (save) model and vectorizer
    load(model, vectorizer, '../models/transaction_classifier_expanded.pkl', '../models/vectorizer_expanded.pkl')

if __name__ == "__main__":
    main()
