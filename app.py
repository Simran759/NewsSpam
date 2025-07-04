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
st.write("Enter a news article **title** below to check if it's fake or real.")

# User input
user_input = st.text_area("📝 Paste your news article title here:")

# Prediction
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news title.")
    else:
        preprocessed = Preprocessing(user_input).text_preprocessing_user()
        vectorized = tfidf.transform(preprocessed)
        prediction = model.predict(vectorized)

        if prediction[0] == 0:
            st.error("🚨 The news is FAKE.")
        else:
            st.success("✅ The news is REAL.")

# Suggested example titles
st.markdown("---")
st.markdown("### 🧪 Try These Example Titles:")
st.markdown("- **Fake News Example:** `Schaeuble to head German parliament, unblocking coalition talks`")
st.markdown("- **Real News Example:** `5 Ways The Media Responded With Butthurt To Donald Trump’s Victory`")
st.markdown("_Paste any of the above into the input box above and click Predict to test your model._")
