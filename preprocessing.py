# imports
# type: ignore
import numpy as np 
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


data = pd.read_csv('data/MBTI500.csv')

def clean_text(text):

    text = text.lower()
    
    # Remove emoticons and emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in word_tokenize(text) if word not in stop_words)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

    # Stemming
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(word) for word in text.split())
    
    return text

data['posts'] = data['posts'].apply(clean_text)

data['type'] = data['type'].str[0]