from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        resume = request.form["resume"]

        vector = vectorizer.transform([resume])

        prediction = model.predict(vector)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)