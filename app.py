import streamlit as st
import pickle
import re
import nltk

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

from preprocessing import Preprocessing

# Load trained model and TF-IDF vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# App UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's fake or real.")

# User input
user_input = st.text_area("Paste your news article here...")

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        preprocessed = Preprocessing(user_input).text_preprocessing_user()
        vectorized = tfidf.transform(preprocessed)
        prediction = model.predict(vectorized)

        if prediction[0] == 0:
            st.error("🚨 The news is FAKE.")
        else:
            st.success("✅ The news is REAL.")
