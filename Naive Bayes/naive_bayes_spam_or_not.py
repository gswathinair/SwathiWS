import pandas as pd

data = pd.read_csv('C:/Users/Rakesh/Desktop/DataScience/Lohith code/Naive Bayes/spam.csv', encoding='latin-1')
data.head()
# Drop column and name change
data = data.drop(["V3", "Unnamed: 3", "Unnamed: 4"], axis=1)

#data2 = data.drop([3,4],axis=0)
#print(data2)

data = data.rename(columns={"v1": "label", "v2": "text"})
data.tail()
data.label.value_counts()
# convert label to a numerical variable
data['label_num'] = data.label.map({'ham':0, 'spam':1})
data.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
vect.fit(X_train)
print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)
type(X_test_df)
ham_words = ''
spam_words = ''
spam = data[data.label_num == 1]
ham = data[data.label_num == 0]
import nltk
import matplotlib.pyplot as plt

for val in spam.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    # tokens = [word for word in tokens if word not in stopwords.words('english')]
    for words in tokens:
        spam_words = spam_words + words + ' '

for val in ham.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '
from wordcloud import WordCloud

# Generate a word cloud image
spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)
# Spam Word cloud
plt.figure(figsize=(10, 8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
# Ham word cloud
plt.figure(figsize=(10, 8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
prediction = dict()
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_df, y_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
prediction["Multinomial"] = model.predict(X_test_df)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, prediction["Multinomial"])
