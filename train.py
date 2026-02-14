import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("AI_Flashcard_Generator/sentiment_nlp/data.csv")

# Split input and output
X = df["text"]
y = df["label"]

# Convert text → numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model + vectorizer
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model saved as sentiment_model.pkl")