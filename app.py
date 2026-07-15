from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        resume = request.form["resume"]

        if resume.strip():
            vector = vectorizer.transform([resume])
            prediction = model.predict(vector)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)