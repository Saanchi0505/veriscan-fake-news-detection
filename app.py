from flask import Flask, render_template, request
from model_utils import detect_fake

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form.get("news")

    if not news_text or news_text.strip() == "":
        return render_template("index.html")

    # Updated to unpack 6 values now
    ml_prediction, rating, source, final_result, method, wiki_topic = detect_fake(news_text)

    if rating:
        fact_result = f"{rating} (Source: {source})"
    else:
        fact_result = "No fact-check data found"

    if wiki_topic:
        wiki_result = f"Wikipedia: {wiki_topic}"
    else:
        wiki_result = "No Wikipedia data found"

    return render_template(
        "index.html",
        ml_prediction=ml_prediction,
        fact_result=fact_result,
        final_result=final_result,
        news_text=news_text,
        method=method,
        wiki_result=wiki_result
    )

if __name__ == "__main__":
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
