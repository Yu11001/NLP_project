import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import streamlit as st
import warnings
warnings.filterwarnings("ignore")


cleaned_data = pd.read_csv('data/cleaned.csv')

cleaned_data.dropna(subset=['posts', 'type'], inplace=True)

cleaned_data['type'] = cleaned_data['type'].map({'I': 0, 'E': 1})

X_train, X_test, y_train, y_test = train_test_split(cleaned_data['posts'], cleaned_data['type'], test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

def predict_mbti(post):
    post_tfidf = vectorizer.transform([post])
    prediction = model.predict(post_tfidf)[0]
    return 'E' if prediction == 1 else 'I'

# Example usage
# user_post = "i always go outside see how world is "
# print(f"Predicted MBTI type: {predict_mbti(user_post)}")

# Streamlit UI
st.title("MBTI Personality Predictor")
st.write("Enter a post to predict whether the author is an Introvert (I) or Extrovert (E)")

user_input = st.text_area("Enter your post here:")
if st.button("Predict"):
    if user_input.strip():
        result = predict_mbti(user_input)
        st.write(f"Predicted MBTI type: {result}")
    else:
        st.write("Please enter a post to get a prediction.")