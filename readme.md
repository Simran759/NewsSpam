# 📰 Fake News Detection using Titles Only

## 🔗 Live Demo  
👉 [Click here to try the Fake News Classifier App](https://newsspam-yutxkyvh7br8fakbeqjwb4.streamlit.app/)

This project is a machine learning-based classifier that detects whether a news article is **fake or real**, using only the **title** of the article. Built with `Random Forest`, the model is optimized for speed and performance using a clean and effective text preprocessing pipeline.

---

## 📌 Why Only Title?

While full article text can provide more information, it:
- 🔁 Takes more **time to preprocess**
- 💾 Is **too large in size**
- 💡 Many fake news headlines are **sensationalized**, so titles alone often carry enough signals to classify accurately.

---

## 🧩 Project Flow

### ✅ 1. Dataset Source

- Extracted from **Kaggle Fake News Dataset**
- Dataset includes: `id`, `title`, `text`, `label`, etc.

---

### 🔍 2. Data Cleaning

- Removed **null values**
- Dropped unnecessary columns (`author`, `text`, etc.)
- Used only:
  - `title`: for input features
  - `label`: target (0 = Fake, 1 = Real)

---

### 🧹 3. Text Preprocessing

Steps applied to each title:
- Tokenization
- Lowercasing
- Stopword removal
- Lemmatization (to reduce words to their base form)
- Joined cleaned tokens into processed text

> These steps help provide meaningful and uniform data to the model.

---

### 🌲 4. Model Choice: Random Forest Classifier

**Why Random Forest?**
- Handles **large datasets** and **sparse input** well
- Reduces **overfitting** through **bagging**
- Performs better than naive models (like Logistic Regression or Naive Bayes) on this classification task

---

### 📊 5. Model Training

- Converted titles into numerical form using **TF-IDF Vectorization**
- Split data into training and testing sets
- Trained using **Random Forest Classifier**
- Evaluated performance

---

### 🎯 6. Results

- Achieved **93% accuracy** on test data
- Very effective even without using full article text

---

## 🖥️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- NLTK (text processing)
- Streamlit (web app deployment)

---

## 🚀 Deployment

Deployed as a **Streamlit Web App** where users can:
- Enter a news title
- Instantly see if it's predicted as **Fake** or **Real**

---

## 📁 Project Structure

📦 NewsSpam/
├── app.py # Streamlit interface
├── model.pkl # Trained Random Forest model
├── tfidf.pkl # TF-IDF vectorizer
├── preprocessing.py # Text cleaning functions
├── requirements.txt # Python dependencies
├── .gitignore # Excludes venv, cache, etc.
└── README.md # This file


---

## 🧠 Final Thought

This project demonstrates that even brief inputs like **titles** can be powerful for fake news detection, especially when paired with the right preprocessing and a robust model like Random Forest.

