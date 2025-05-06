from flask import Flask, render_template, request
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import Article

# Initialize Flask
app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define political keywords
left_keywords = ["climate change", "gun control", "abortion rights", "racial equity", "lgbtq", "universal healthcare", "income inequality", "environmental justice"]
right_keywords = ["border security", "second amendment", "pro life", "tax cuts", "patriotism", "traditional values", "religious freedom", "illegal immigration"]
center_keywords = ["bipartisan", "moderate", "neutral stance", "independent", "nonpartisan", "centrist"]

def extract_keywords(text):
    found = {"left": [], "right": [], "center": []}
    lower_text = text.lower()
    for kw in left_keywords:
        if kw in lower_text:
            found["left"].append(kw)
    for kw in right_keywords:
        if kw in lower_text:
            found["right"].append(kw)
    for kw in center_keywords:
        if kw in lower_text:
            found["center"].append(kw)
    return found

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    chart_url = None

    if request.method == "POST":
        url = request.form.get("url")
        manual_text = request.form.get("manual_text")

        article_text = ""
        source = "Manual Input"

        try:
            if url:
                article = Article(url)
                article.download()
                article.parse()
                article_text = article.text
                source = url
            elif manual_text.strip():
                article_text = manual_text
            else:
                result = {"error": "Please provide either article text or a URL."}
                return render_template("index.html", result=result)

            # Predict with model
            X_input = vectorizer.transform([article_text])
            pred = model.predict(X_input)[0]
            probs = model.predict_proba(X_input)[0]

            # Generate bar chart
            fig, ax = plt.subplots()
            ax.bar(range(len(probs)), probs)
            ax.set_xticks(range(len(probs)))
            ax.set_xticklabels(["Right", "Lean Right", "Center", "Lean Left", "Left"])
            ax.set_ylabel("Confidence")
            ax.set_title("Political Bias Prediction Probabilities")

            chart_path = os.path.join("static", "chart.png")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()

            # Keyword extraction
            keywords = extract_keywords(article_text)

            label_map = {
                0: "Right",
                1: "Lean Right",
                2: "Center",
                3: "Lean Left",
                4: "Left"
            }

            result = {
                "bias": label_map.get(pred, "Unknown"),
                "keywords": keywords,
                "snippet": article_text[:500] + ("..." if len(article_text) > 500 else ""),
                "source": source
            }

            chart_url = chart_path

        except Exception as e:
            result = {"error": f"An error occurred: {str(e)}"}

    return render_template("index.html", result=result, chart_url=chart_url)

if __name__ == "__main__":
    app.run(debug=True)
