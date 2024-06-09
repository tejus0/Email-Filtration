# %%
import pandas as pd

#import requests
#url = 'https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv'
#res = requests.get(url, allow_redirects=True)
#with open('sales_team.csv','wb') as file:
 #   file.write(res.content)
#sales_team = pd.read_csv('sales_team.csv')

df = pd.read_csv('../naive buyes/spam mail.csv')
df.head()

# %%
df.groupby('Category').describe()  # grouping ham and spam 

# %%
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)  # marking ham =0 and spam =1
df.head()

# %%
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(df.Message,df.spam,test_size=0.25)

# %%
#extracting features from text 

from sklearn.feature_extraction.text import CountVectorizer

v= CountVectorizer()
X_train_count = v.fit_transform(X_train.values) # converting text data into matrix of word count
#X_train_count.toarray()  # encoding text features in integer indexes
X_train_count.toarray()[:3]

# %%
# Multinomial bayes is used as we are dealing with discrete data like frequency of words in text .
from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB()
model.fit(X_train_count, y_train)  

# %%
# Prediction of model 
emails =[
    'Hey how are you ?' ,
    'Upto 20% off on sale of laptops, exclusive offer !  '
]
email_count=v.transform(emails)
predictions = model.predict(email_count)

for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Email '{emails[i]}' is spam")
    else:
        print(f"Email '{emails[i]}' is ham")


# %%
# Evaluating the model
X_test_count = v.transform(X_test.values)
y_pred = model.predict(X_test_count)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
