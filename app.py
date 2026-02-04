import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ---------- UI ----------
st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠")

st.title("🧠 Sentiment Analysis App")
st.write("Type a sentence and click **Analyze**")

user_input = st.text_area("Enter text:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction.lower() == "positive":
            st.success("😊 Sentiment: POSITIVE")
        else:
            st.error("😞 Sentiment: NEGATIVE")
