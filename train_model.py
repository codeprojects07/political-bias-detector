import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

# Load dataset (make sure this file exists in your project folder)
df = pd.read_csv("political_bias.csv")

# Check and rename necessary columns if needed
df = df.rename(columns={"text": "text", "label": "label"})  # adjust if column names differ

# Split the data
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline.named_steps["clf"], f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(pipeline.named_steps["tfidf"], f)
