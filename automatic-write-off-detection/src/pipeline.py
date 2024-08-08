import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import openai
import joblib

import yaml
from utils import load_env_vars
from logger import logger

from categorization import categorize_transaction
from merchant_identification import extract_entities, prioritize, find_best_match

openai_api_key = load_env_vars()
openai.api_key = openai_api_key

def extract(file_path):
    '''
    Extract data from a CSV file

    :param file_path: Path to the CSV file
    :return: DataFrame with extracted data
    '''
    logger.info(f"Extracting data from {file_path}")
    return pd.read_csv(file_path)

def gpt_feature_extraction(transaction):
    '''
    Extract features from a transaction description using OpenAI's GPT model

    :param transaction: Transaction description
    :return: Extracted features
    '''
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "extract features from this transaction description."},
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
    logger.info("Transforming data")
    df['plaid_merchant_description'] = df['plaid_merchant_description'].fillna('')

    df['entities'] = df['plaid_merchant_description'].apply(lambda x: extract_entities(x))
    df['merchant'] = df['entities'].apply(lambda x: prioritize(x)).fillna('')

    df['transaction_type'] = df['entities'].apply(lambda x: x['transaction_type']).fillna('')
    df['amount'] = df['entities'].apply(lambda x: x['amount']).fillna('')
    df['plaid_category'] = df['entities'].apply(lambda x: x['plaid_category']).fillna('')

    df['gpt_features'] = df['plaid_merchant_description'].apply(gpt_feature_extraction)

    df['category'] = df['plaid_merchant_description'].apply(categorize_transaction)

    # find duplicates and remove
    df = df.drop_duplicates(subset=['plaid_merchant_description', 'keeper_merchant_description'])

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
    logger.info(f"Training model: {model_type}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'linear_regression':
        model = LogisticRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #classification report
    display_classification_report(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    return model


def display_classification_report(y_test, y_pred):
    '''
    Display classification report in readable format

    :param y_test: True target values
    :param y_pred: Predicted target values.
    '''
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    report_df = pd.DataFrame(report_dict).transpose()

    logger.info("\nClassification Report:")
    print(report_df)


def load(model, vectorizer, model_path, vectorizer_path):
    '''
    Save trained model and vectorizer to disk

    :param model: Trained model
    :param vectorizer: TF-IDF vectorizer
    :param model_path: File path to save the model
    :param vectorizer_path: File path to save the vectorizer
    '''
    logger.info(f"Saving model to {model_path} and vectorizer to {vectorizer_path}")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def main():
    '''
    Main function that extract data, transform, trains model, and save model
    '''
    #load
    with open("../config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    #extract
    df = extract(config['data_path'])

    #transform
    vectorizer = TfidfVectorizer()
    X, y = transform(df, vectorizer)

    #train model
    model = train_model(X, y, model_type='naive_bayes')

    #load (save) model and vectorizer
    load(model, vectorizer, config['model_path'], config['vectorizer_path'])

if __name__ == "__main__":
    main()
