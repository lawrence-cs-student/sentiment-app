import pandas as pd
import nltk
import string
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


nltk.download("punkt")
nltk.download("punkt_tab") 
nltk.download("stopwords")


df = pd.read_csv("sentiment_dataset.csv")


if "text" not in df.columns or "sentiment" not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'sentiment' columns")

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


df["clean_text"] = df["text"].astype(str).apply(preprocess)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully.")
