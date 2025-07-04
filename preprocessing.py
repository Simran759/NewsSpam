import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class Preprocessing:
    def __init__(self, data):
        self.data = data

    def text_preprocessing_user(self):
        lm = WordNetLemmatizer()
        pred_data = [self.data]
        cleaned_data = []

        for data in pred_data:
            review = re.sub('[^a-zA-Z]', ' ', data)
            review = review.lower().split()
            review = [lm.lemmatize(word) for word in review if word not in stop_words]
            cleaned_data.append(' '.join(review))

        return cleaned_data
