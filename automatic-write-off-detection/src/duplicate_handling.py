import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#sample data
data = {
    'info': [
        'PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX',
        'PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX',
        'BP PRODUCTS NORTH AMERICA'
    ],
    'is_duplicate': [0, 1, 0]
}
df = pd.DataFrame(data)

#turn into vector
vectorizer = TfidfVectorizer()
info_data = vectorizer.fit_transform(df['info'])
cat_data = df['is_duplicate']

#train and test sets
info_data_train, info_data_test, cat_data_train, cat_data_test = train_test_split(info_data, cat_data, test_size=0.2,
                                                                                  random_state=42)
#Logistic Regression
model = LogisticRegression()
model.fit(info_data, cat_data)


def find_duplicate(transaction1,transaction2):
    new_info_data = vectorizer.transform([transaction1,transaction2])
    return model.predict(new_info_data)

#test
if __name__ == "__main__":
    new_transactions = [
        'PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX',
        'BP PRODUCTS NORTH AMERICA'
    ]
    duplicates = find_duplicate(*new_transactions)
    print(f"Transaction: {new_transactions}")
    if duplicates[0] == 1:
        print("Yes")
    else:
        print("No")

'''
Transaction: ['PURCHASE BP#8773269DE MOUNT JULIET TN CARDXXXX', 'BP PRODUCTS NORTH AMERICA']
No
'''