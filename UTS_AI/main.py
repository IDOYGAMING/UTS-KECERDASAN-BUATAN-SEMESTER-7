import pandas as pd
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

data = pd.read_csv("dataset.csv")

data = data[["tweet", "class"]]
data["label"] = data["class"].replace({0:1, 1:1, 2:0})
data = data.rename(columns={"tweet": "text"})
data = data[["text","label"]]

print("Contoh data:")
print(data.head())

stop_words = set(stopwords.words("indonesian"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data["clean_text"] = data["text"].apply(clean_text)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data["clean_text"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

data["label"].value_counts().plot(kind="bar")
plt.title("Distribusi Label")
plt.xticks(ticks=[0,1], labels=["Tidak Kebencian (0)","Kebencian (1)"])
plt.show()

neutral_text = " ".join(data[data["label"]==0]["clean_text"])
wordcloud_neutral = WordCloud(width=800, height=400, background_color="black").generate(neutral_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Neutral labels")
plt.show()

hate_text = " ".join(data[data["label"]==1]["clean_text"])
wordcloud_hate = WordCloud(width=800, height=400, background_color="black").generate(hate_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_hate, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Hate Speech labels")
plt.show()
