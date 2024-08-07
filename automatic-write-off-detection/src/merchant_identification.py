import re
from fuzzywuzzy import fuzz, process


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
        merchant = identify_merchant(t, sample_t)

        print(f"Transaction: {t}")
        print(f"Entities: {entities}")
        print(f"Merchant: {merchant}")
        print("-" * 40)
