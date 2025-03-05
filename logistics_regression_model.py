import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")

# Load data
cleaned_data = pd.read_csv('data/cleaned.csv')
cleaned_data.dropna(subset=['posts', 'type'], inplace=True)
cleaned_data['type'] = cleaned_data['type'].map({'I': 0, 'E': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_data['posts'], cleaned_data['type'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear'], 'class_weight': [None, 'balanced']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

# Best model
model = grid_search.best_estimator_

# Save model and vectorizer
joblib.dump(model, 'mbti_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))