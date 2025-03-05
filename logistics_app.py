import streamlit as st
import joblib

# Load Model and Vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('mbti_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Function to Predict MBTI
@st.cache_data
def transform_text(text):
    return vectorizer.transform([text])

def predict_mbti(post):
    post_tfidf = transform_text(post)
    prediction = model.predict(post_tfidf)[0]
    return 'E' if prediction == 1 else 'I'

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
