import streamlit as st
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Load dataset
cleaned_data = pd.read_csv('data/cleaned.csv')
# Drop missing values in important columns
cleaned_data.dropna(subset=['posts', 'type'], inplace=True)
cleaned_data=cleaned_data.head(2000)
# Convert MBTI types to binary classification (assuming only 'I' and 'E' exist)
cleaned_data['type'] = cleaned_data['type'].map({'I': 0, 'E': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(cleaned_data['posts'], cleaned_data['type'], test_size=0.6, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)

#apply SMOTE on training data
undersampler = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_tfidf, y_train)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=7,metric='euclidean',weights='uniform',algorithm='auto')
knn.fit(X_train_balanced, y_train_balanced)

# Model evaluation
y_pred = knn.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.4f}")
print(f"Classification report: {classification_report(y_test,y_pred)}")

# Function to predict MBTI type
def predict_mbti(post):
    post_tfidf = vectorizer.transform([post])
    prediction = knn.predict(post_tfidf)[0]
    return 'Extrovert' if prediction == 1 else 'Introvert'

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
        