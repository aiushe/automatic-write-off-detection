import re
from fuzzywuzzy import fuzz, process
import pandas as pd
from rank_bm25 import BM25Okapi

def load_merchant_entities(file_path):
    """
    Load merchant entities from a CSV file

    :param file_path: path to CSV file
    :return: dataFrame with merchant entities
    """
    return pd.read_csv(file_path)

merchant_df = load_merchant_entities('../data/merchant_entities.csv')
corpus = merchant_df['merchant_name'].tolist()
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def extract_entities(transaction):
    '''
    Extract entities from a transaction description using regular expressions

    :param transaction: transaction description
    :return: dictionary that contains the extracted entities
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

    :param entities: dictionary that contains the extracted entities
    :return: prioritized entity or 'UNKNOWN'
    '''
    if entities['merchant']:
        return entities['merchant']
    elif entities['transaction_type']:
        return entities['transaction_type']
    else:
        return 'UNKNOWN'


def bm25_search(query, corpus):
    """
    Perform BM25 search on the corpus

    :param query: query string
    :param corpus: list of merchants
    :return: document, score sorted by score
    """
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    return sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)


def find_best_match(query, merchant_df, threshold=75):
    """
    Find the best match for query using BM25 and fuzzy matching.

    :param query: query string
    :param merchant_df: dataFrame with merchant entities
    :param threshold: fuzzy matching score
    :return: best matching merchant name and category
    """
    corpus = merchant_df['merchant_name'].tolist()
    bm25_results = bm25_search(query, corpus)
    top_result = bm25_results[0][0]  # Get the top result from BM25
    fuzzy_score = fuzz.token_sort_ratio(query, top_result)

    if fuzzy_score >= threshold:
        category = merchant_df[merchant_df['merchant_name'] == top_result]['category'].values[0]
        return top_result, category
    else:
        return 'UNKNOWN', 'UNKNOWN'

if __name__ == "__main__":
    sample_t = [
        'PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX',
        'DEBIT CARD DEBIT / AUTH #710620 03-01-2023 BP#5364328RPF',
        'BP PRODUCTS NORTH AMERICA',
        'BP AFFIRM * PAY Y8QBBI0A',
        'PRE-AUTHORIZATION DEBIT AT BP#95, CALUMET CITY, IL FROM CARD#: XXXXXX',
        'POS DEBIT- 4059 4059 BP#8982183BP AMPM',
        'BP#8543514BP KENOSHA T',
        'PURCHASE AUTHORIZED ON 02/28 BP#9704412MT ZION ROAD B MORROW GA P383060090732083'
    ]

    for t in sample_t:
        entities = extract_entities(t)
        merchant, category = find_best_match(t, sample_t)

        print(f"Transaction: {t}")
        print(f"Entities: {entities}")
        print(f"Merchant: {merchant}")
        print(f"Category: {category}")
        print("-" * 40)
