# 📰 Fake News Title Classifier

A machine learning web application that detects whether a news **title** is fake or real using **NLP** and a **Random Forest classifier**, deployed with **Streamlit**.

🔗 **Live App**: [Streamlit Deployment](https://newsspam-yutxkyvh7br8fakbeqjwb4.streamlit.app)

---

## 🚀 Overview

This project focuses on building a lightweight and efficient fake news detection system using **only the news headline (title)**. The model is trained on a labeled dataset and achieves **93% accuracy** using a Random Forest classifier and TF-IDF vectorization.

---

## 🧠 Why Only Title?

- Processing full article content is time-consuming and heavy for real-time applications.
- News titles often contain emotional or misleading cues, sufficient for classification.
- Improved deployment speed and reduced model complexity.

---

## 🧪 Model & Approach

- **Dataset**: Kaggle Fake News Dataset
- **Goal**: Binary Classification → Fake / Real
- **Model**: `RandomForestClassifier` from scikit-learn
- **Vectorizer**: TF-IDF on `title` column only

### 🔍 Why Random Forest?

- Effective with sparse high-dimensional data like TF-IDF.
- Resistant to overfitting (ensemble of decision trees).
- Fast training and prediction, good accuracy.
- Intuitive and interpretable results.

---

## ⚙️ Preprocessing Pipeline

1. Dropped null entries and unused columns
2. Cleaned text with:
   - Regex removal of non-alphabetic chars
   - Lowercasing
   - Stopwords removal
   - Lemmatization using NLTK
3. Rejoined words into clean title text
4. Applied `TfidfVectorizer` for feature extraction

---

## 📊 Performance

| Metric      | Result  |
|-------------|---------|
| Accuracy    | 93%     |
| Precision   | High    |
| Inference   | Near Instant |
| Data Used   | Titles Only (10,000 rows)

---

## 🧠 Learnings

✅ **Applied TF-IDF for feature extraction**, understanding the importance of term weighting in text classification.  
✅ **Implemented Random Forest** for robust binary classification with high-dimensional input.  
✅ **Learned how to handle real-world datasets**: cleaning, reducing dimensionality, avoiding overfitting.  
✅ Understood the tradeoff between **model complexity** and **speed** in deployment scenarios.  
✅ Gained **hands-on practice with deployment using Streamlit**, improving understanding of real-time ML apps.  
✅ Practiced **writing reusable components** (e.g., `Preprocessing` class) to ensure cleaner, scalable code.  
✅ Explored how to convert notebook-based work into a modular and **production-ready app**.  
✅ Gained insights into model bias (e.g., class imbalance) and how to evaluate it with real user feedback.

---

## 📁 Project Structure

```
├── app.py # Streamlit web application
├── preprocessing.py # Preprocessing functions (tokenizing, lemmatizing, cleaning)
├── model.pkl # Trained Random Forest model (pickle)
├── tfidf.pkl # TF-IDF vectorizer object
├── requirements.txt # List of required Python packages
└── README.md # Project documentation
```