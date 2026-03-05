import pandas as pd
import numpy as np
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# 2️⃣ Add labels
fake["label"] = 0   # Fake = 0
true["label"] = 1   # Real = 1

# 3️⃣ Combine datasets
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# 4️⃣ Keep only text and label
data = data[["text", "label"]]

# 5️⃣ Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

data["text"] = data["text"].apply(clean_text)

# 6️⃣ Split data
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 8️⃣ Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 9️⃣ Test accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# 🔟 Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")