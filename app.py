import streamlit as st
import pickle
import re
from preprocessing import Preprocessing
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# Load model and TF-IDF
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's fake or real.")

user_input = st.text_area("Paste your news article here...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        preprocessed = Preprocessing(user_input).text_preprocessing_user()
        vectorized = tfidf.transform(preprocessed)
        prediction = model.predict(vectorized)

        if prediction[0] == 0:
            st.error("🚨 The news is FAKE.")
        else:
            st.success("✅ The news is REAL.")
