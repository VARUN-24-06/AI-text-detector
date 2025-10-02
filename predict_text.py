import joblib
import re
import nltk
import numpy as np
import textstat
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load models
tfidf_vector = joblib.load("tfidf.pkl")
svd = joblib.load("svd.pkl")
booster = joblib.load("xgb_booster.pkl")
log_reg = joblib.load("log_reg.pkl")

# Setup
REPLACE_BAD_WORD = re.compile(r'[/(){}\[\]\|@,;]')
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = REPLACE_BAD_WORD.sub(" ", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def extract_linguistic_features(text):
    readability = textstat.flesch_reading_ease(text)
    words = text.split()
    vocab_div = len(set(words)) / len(words) if words else 0
    avg_sent_len = np.mean([len(s.split()) for s in text.split('.') if s.strip()]) if '.' in text else len(words)
    return np.array([readability, vocab_div, avg_sent_len])

def predict_text(text):
    cleaned = clean_text(text)
    vector = tfidf_vector.transform([cleaned])
    reduced = svd.transform(vector)
    ling = extract_linguistic_features(text).reshape(1, -1)
    final_features = np.hstack([reduced, ling])

    # Predict with XGBoost
    dmatrix = xgb.DMatrix(final_features)
    xgb_proba = booster.predict(dmatrix)

    # Predict with Logistic Regression
    lr_proba = log_reg.predict_proba(final_features)[:, 1]

    # Soft voting
    avg_proba = (xgb_proba + lr_proba) / 2
    ai_score = avg_proba[0] * 100
    human_score = 100 - ai_score

    # Uncertainty zone
    if 40 <= ai_score <= 60:
        label = "Uncertain"
    elif ai_score > 60:
        label = "AI"
    else:
        label = "Human"

    # Readability metrics
    readability, vocab_div, avg_sent_len = ling[0]

    print(f"\nPrediction: {label}")
    print(f"Human: {human_score:.2f}% | AI: {ai_score:.2f}%")
    print(f"Readability: {readability:.2f}")
    print(f"Vocabulary Diversity: {vocab_div:.2f}")
    print(f"Average Sentence Length: {avg_sent_len:.2f} words")

# Example usage
if __name__ == "__main__":
    text = input("Enter text to classify: ")
    predict_text(text)
