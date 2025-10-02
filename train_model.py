import pandas as pd
import numpy as np
import re
import nltk
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import textstat

# NLTK setup
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------
# 1. Load & Clean Dataset
# -----------------------
start = time.time()
data = pd.read_csv("C:/Users/Sudhan/Downloads/archive/AI_Human.csv")

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

data["clean_text"] = data["text"].apply(clean_text)

train, temp = train_test_split(data, test_size=0.4, stratify=data["generated"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["generated"], random_state=42)

x_train, y_train = train["clean_text"].values, train["generated"].values
x_val, y_val = val["clean_text"].values, val["generated"].values
x_test, y_test = test["clean_text"].values, test["generated"].values
print("✅ Data loaded and cleaned in", round(time.time() - start, 2), "seconds")

# -----------------------
# 2. TF-IDF + SVD
# -----------------------
start = time.time()
tfidf_vector = TfidfVectorizer(max_features=10000, min_df=10, max_df=0.85, ngram_range=(1,1))
tfidf_vector.fit(x_train)

x_train_tfidf = tfidf_vector.transform(x_train)
x_val_tfidf   = tfidf_vector.transform(x_val)
x_test_tfidf  = tfidf_vector.transform(x_test)

svd = TruncatedSVD(n_components=150, random_state=42)
x_train_svd = svd.fit_transform(x_train_tfidf)
x_val_svd   = svd.transform(x_val_tfidf)
x_test_svd  = svd.transform(x_test_tfidf)
print("✅ TF-IDF + SVD completed in", round(time.time() - start, 2), "seconds")

# -----------------------
# 3. Linguistic Features
# -----------------------
start = time.time()
def extract_linguistic_features(texts):
    features = []
    for t in texts:
        readability = textstat.flesch_reading_ease(t)
        words = t.split()
        vocab_div = len(set(words)) / len(words) if words else 0
        avg_sent_len = np.mean([len(s.split()) for s in t.split('.') if s.strip()]) if '.' in t else len(words)
        features.append([readability, vocab_div, avg_sent_len])
    return np.array(features)

x_train_ling = extract_linguistic_features(x_train)
x_val_ling   = extract_linguistic_features(x_val)
x_test_ling  = extract_linguistic_features(x_test)
print("✅ Linguistic features extracted in", round(time.time() - start, 2), "seconds")

# Combine features
x_train_final = np.hstack([x_train_svd, x_train_ling])
x_val_final   = np.hstack([x_val_svd, x_val_ling])
x_test_final  = np.hstack([x_test_svd, x_test_ling])

# -----------------------
# 4. Train Models (Native XGBoost)
# -----------------------
start = time.time()
dtrain = xgb.DMatrix(x_train_final, label=y_train)
dval   = xgb.DMatrix(x_val_final, label=y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42
}

booster = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    early_stopping_rounds=20,
    evals=[(dval, "eval")],
    verbose_eval=True
)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=300)
log_reg.fit(x_train_final, y_train)
print("✅ Models trained in", round(time.time() - start, 2), "seconds")

# -----------------------
# 5. Save Models
# -----------------------
joblib.dump(tfidf_vector, "tfidf.pkl")
joblib.dump(svd, "svd.pkl")
joblib.dump(booster, "xgb_booster.pkl")
joblib.dump(log_reg, "log_reg.pkl")

print("✅ Training complete. Models saved.")
