import joblib

# Load the saved model and vectorizer
model = joblib.load("resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sample resume
resume = """
Python developer with experience in Django, Flask,
Machine Learning, SQL, Pandas and NumPy.
"""

# Convert resume into TF-IDF features
resume_vector = vectorizer.transform([resume])

# Predict category
prediction = model.predict(resume_vector)

print("Predicted Category:")
print(prediction[0])