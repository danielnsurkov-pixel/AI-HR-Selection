import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack

resumes = pd.read_csv("data/resumes.csv")
with open("data/candidates.json", encoding="utf-8") as f:
    candidates = json.load(f)
with open("data/stopwords.txt", encoding="utf-8") as f:
    stopwords = [w.strip() for w in f.readlines()]

candidates_df = pd.DataFrame(candidates)
merged = resumes.merge(candidates_df, on="id", suffixes=("_resume", "_cand"))

vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=500)
X_text = vectorizer.fit_transform(merged["text"])

mlb = MultiLabelBinarizer()
X_skills = mlb.fit_transform(merged["skills"])

X = hstack([X_text, X_skills])
y = merged["label_resume"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro", zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average="macro", zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, average="macro", zero_division=0))

