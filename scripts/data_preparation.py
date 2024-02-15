import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from .settings import BASE_PATH
import os

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

print(BASE_PATH)

def load_vectorizer():
    vectorizer = joblib.load(os.path.join(BASE_PATH, '../model/vectorizer.joblib'))
    return vectorizer

def load_scaler():
    scaler = joblib.load(os.path.join(BASE_PATH, '../model/scaler.joblib'))
    return scaler

def load_model():
    model = joblib.load(os.path.join(BASE_PATH, '../model/model.joblib'))
    return model

def clean_text(text):
    # Remove special characters
    text = text.replace('Ã¯Â¿Â½', '')
    
    # Lower casing
    text = text.lower()
    
    # Remove mentions and links
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'http\S+|bit.ly\S+', '', text)

    return text

def preprocess_text(text):
    # Remove punctuations
    text = re.sub(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", '', text)

    # Remove numbers
    text = re.sub(r'([0-9]+)', '', text)

    # Remove more than one space
    text = re.sub(r' +', ' ', text)
    # Trim the text
    text = text.strip()

    # Remove stopwords and tokenize then join
    text = [token for token in word_tokenize(text) if token not in stop_words]
    text = ' '.join(text)

    return text

vectorizer = load_vectorizer()
scaler = load_scaler()

def extract_features(text):
    features = vectorizer.transform(text)
    features = scaler.transform(features)
    return features

model = load_model()

def make_prediction(features):
    predictions = model.predict(features)
    return predictions