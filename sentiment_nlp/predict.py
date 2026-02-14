import pickle

# Load saved model and vectorizer
with open("AI_Flashcard_Generator/sentiment_nlp/sentiment_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# Take user input
text = input("Enter a sentence: ")

# Convert text → numbers
text_vector = vectorizer.transform([text])

# Predict sentiment
prediction = model.predict(text_vector)

print("Sentiment:", prediction[0])