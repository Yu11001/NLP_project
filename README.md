# NLP Project: MBTI Personality Prediction

## 📌 Project Overview

This project aims to predict whether 'Extrovert' or 'Introvert' personality types based on posts and text records using Natural Language Processing (NLP). It leverages 2 machine learning and 1 deep learning techniques to analyze text and classify personality types.

## 🚀 Features

- **Personality Classification**: Predicts personality types based on text input.
- **Machine Learning Models**: Utilizes NLP techniques for feature extraction and classification.
- **Web Interface**: Displays results interactively.

## 📂 Project Structure

```
├── data/                                 # Contains datasets
│   ├── MBTI500.csv                       # original dataset
│   ├── cleaned.csv                       # cleaned dataset after running `preprocessing.py`
├── models/                               # Trained models
│   ├── logistics_regression_model.py     # LR model
├── pkl/                                  # to store the pkl file
│   ├── mbti_model.pkl                    # Data cleaning and preparation
│   ├── tfidf_vectorizer.pkl              # Machine learning model
├── app.py                                # combine 3 models
├── logistics_app.py                      # LR model interface
├── preprocessing.py                      # preprocessing file
├── requirements.txt                      # Required Python packages
└── README.md                             # This file
```

## 📥 Dataset

`data/MBTI500.csv`: original dataset

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Yu11001/NLP_project.git
   cd NLP_project
   python preprocessing.py
   python models/logistics_regression_model.py
   pip install streamlit #in case your environment doesn't have it
   streamlit run logistics_app.py # for running logistics_regression model
   ```

## 🛠 Technologies Used

- **Python** (Pandas, Scikit-Learn, NLTK, Transformers)
- **Jupyter Notebooks** (for experimentation)
- **Git & GitHub** (for version control)

## 📢 Contributing

Feel free to submit issues and pull requests. Contributions are welcome!

## 📜 License

MIT License
