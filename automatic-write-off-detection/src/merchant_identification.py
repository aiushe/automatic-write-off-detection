import re
from fuzzywuzzy import fuzz, process


def extract_entities(transaction):
    #disect transaction data following the pattern (MetaData, Bank, App) and ignore the rest
    pattern = {
        'metadata': r'(DEBIT WITHDRAWAL|PURCHASE|PRE-AUTHORIZATION DEBIT|POS DEBIT)',
        'bank': r'(WELLS FARGO|BANK OF AMERICA|CHASE|CITI|BP)',
        'app': r'(UBER|LYFT|PAYPAL|AFFIRM|APPLE)'
    }

    entities = {'metadata': None, 'bank': None, 'app': None}

    #loop throught the string to find and determine the important info defined in the pattern above
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

#test
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
        '''
        output
            Transaction: PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX
            Entities: {'metadata': 'PURCHASE', 'bank': 'CHASE', 'app': None}
            Merchant: PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX
        '''
