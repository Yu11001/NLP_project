import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset from CSV
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned.csv")  # Ensure the path is correct

# Train model and cache it
@st.cache_data
def train_model():
    cleaned_data = load_data()
    
    X = cleaned_data['posts']
    y = cleaned_data['type']

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Model evaluation
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.4f}")

    return knn, vectorizer

# Function to predict MBTI type
def predict_mbti(post, model, vectorizer):
    post_tfidf = vectorizer.transform([post])  # Convert text to TF-IDF features
    probabilities = model.predict_proba(post_tfidf)[0]  # Get prediction probabilities
    
    result = 'I' if probabilities[0] > probabilities[1] else 'E'
    return result, probabilities

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['I', 'E'], yticklabels=['I', 'E'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Streamlit UI
def main():
    st.title("MBTI Personality Predictor")
    st.write("Enter a post to predict whether the author is an Introvert (I) or Extrovert (E)")

    # Load the model and vectorizer
    model, vectorizer = train_model()

    # Input text for prediction
    user_input = st.text_area("Enter your post here:")
    if st.button("Predict"):
        if user_input.strip():  # Check if input is not empty
            result, probabilities = predict_mbti(user_input, model, vectorizer)
            st.write(f"Predicted MBTI type: **{result}**")

            # Display prediction probabilities
            st.write("### Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['Introvert (I)', 'Extrovert (E)'],
                'Probability': probabilities
            })
            st.bar_chart(prob_df.set_index('Class'))
        else:
            st.write("Please enter a post to get a prediction.")

if __name__ == "__main__":
    main()
