import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

#sample data for training
data = {
    'info': [
        'PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX',
        'DEBIT CARD DEBIT / AUTH #710620 03-01-2023 BP#5364328RPF',
        'BP PRODUCTS NORTH AMERICA',
        'BP AFFIRM * PAY Y8QBBI0A',
        'PRE-AUTHORIZATION DEBIT AT BP#95, CALUMET CITY, IL FROM CARD#: XXXXXX',
        'POS DEBIT- 4059 4059 BP#8982183BP AMPM',
        'BP#8543514BP KENOSHA T',
        'PURCHASE AUTHORIZED ON 02/28 BP#9704412MT ZION ROAD B MORROW GA P383060090732083'
    ],
    'category': ['Purchase', 'Debit', 'Purchase', 'Payment', 'Debit', 'POS', 'Purchase', 'Purchase']
}
df = pd.DataFrame(data)

#turn into vector
vectorizer = TfidfVectorizer()
info_data = vectorizer.fit_transform(df['info'])
cat_data = df['category']

#train and test sets
info_data_train, info_data_test, cat_data_train, cat_data_test = train_test_split(info_data, cat_data, test_size=0.2,
                                                                                  random_state=42)

#Navie Bayes
model = MultinomialNB()
model.fit(info_data, cat_data)

joblib.dump(model, '../models/transaction_classifier.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')