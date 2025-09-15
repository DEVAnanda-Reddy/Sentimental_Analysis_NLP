"""
train_sentiment_model.py
Production-ready training script for sentiment analysis.
Saves both the TF-IDF vectorizer and the trained classifier.
"""

import re
import string
import pickle
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ========================
# 1. Setup
# ========================
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
data_path = "Reviews.csv"   # <-- ensure your dataset is here
df = pd.read_csv(data_path)

# ========================
# 2. Sampling (optional)
# ========================
df = df.sample(10000, random_state=42)   # remove this line for full training

# ========================
# 3. Label creation
# ========================
df["SentimentPolarity"] = df["Score"].apply(
    lambda x: "Positive" if x > 3 else ("Negative" if x < 3 else "Neutral")
)
df["Class"] = df["SentimentPolarity"].apply(
    lambda x: 2 if x == "Positive" else (0 if x == "Negative" else 1)
)

# Remove invalid rows
df = df[df["HelpfulnessNumerator"] <= df["HelpfulnessDenominator"]]

# Drop unused columns
df = df.drop([
    "Id","ProductId","Time","SentimentPolarity","UserId",
    "Score","ProfileName","HelpfulnessNumerator",
    "HelpfulnessDenominator","Summary"
], axis=1)

# ========================
# 4. Preprocessing functions
# ========================
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def to_lowercase(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def clean_html(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)

def preprocess_text(text):
    text = remove_punctuation(str(text))
    text = to_lowercase(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = clean_html(text)
    return text

# Apply preprocessing
df["cleaned_text"] = df["Text"].apply(preprocess_text)
df = df.drop(["Text"], axis=1)

# ========================
# 5. Train-test split
# ========================
X = df["cleaned_text"]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# 6. TF-IDF + Logistic Regression
# ========================
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression(C=10, solver="saga", max_iter=5000)
model.fit(X_train_tfidf, y_train)

# ========================
# 7. Evaluation
# ========================
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# ========================
# 8. Save model + vectorizer
# ========================
with open("sentiment_model_tfidf.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print("🎉 Training complete. Saved sentiment_model_tfidf.pkl and vectorizer_tfidf.pkl")
