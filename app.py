from flask import Flask, render_template, request
from newspaper import Article
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('punkt')

app = Flask(__name__)

# Political bias phrases
bias_phrases = {
    "left": [
        "climate crisis", "racial justice", "gun control", "universal healthcare", "systemic racism",
        "environmental justice", "income inequality", "trans rights", "abortion access", "defund the police"
    ],
    "right": [
        "border crisis", "illegal immigration", "second amendment rights", "tax cuts", "traditional values",
        "cancel culture", "leftist agenda", "pro life", "religious freedom", "fake news media"
    ]
}

def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

def detect_bias_phrases(text, side):
    matches = []
    for phrase in bias_phrases[side]:
        pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
        if re.search(pattern, text):
            matches.append(phrase)
    return matches

def classify_bias(text):
    left_hits = detect_bias_phrases(text, "left")
    right_hits = detect_bias_phrases(text, "right")

    if len(left_hits) > len(right_hits):
        return "Left-Leaning", left_hits
    elif len(right_hits) > len(left_hits):
        return "Right-Leaning", right_hits
    else:
        return "Neutral", []

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form.get("url")
        manual_text = request.form.get("manual_text")

        if url:
            article_text = extract_article_text(url)
            source = "URL"
        elif manual_text:
            article_text = manual_text
            source = "Manual"
        else:
            article_text = None
            source = "None"

        if article_text:
            bias_label, matched_phrases = classify_bias(article_text)
            sentiment = analyze_sentiment(article_text)
            snippet = article_text[:500] + "..." if len(article_text) > 500 else article_text

            result = {
                "bias": bias_label,
                "sentiment": f"{sentiment:.2f}",
                "phrases": matched_phrases,
                "snippet": snippet,
                "source": source
            }
        else:
            result = {"error": "Please enter a valid article URL or paste the article text."}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
