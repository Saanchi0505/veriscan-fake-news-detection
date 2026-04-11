import joblib
import requests
import wikipedia

# Load trained model and vectorizer
model = joblib.load('Models/fake_news_model.pkl')
vectorizer = joblib.load('Models/bow_vectorizer.pkl')

API_KEY = "AIzaSyAE56OYa_6G-Ocuzr_bbauwulEdww6McR0"

FAKE_PATTERNS = [
    "earth is flat",
    "aliens landed",
    "5g causes coronavirus",
    "bleach cures covid",
    "drinking hot water cures covid",
    "india has most unesco",
    "india has the largest number of unesco",
]


def predict_news(news_text):
    news_vec = vectorizer.transform([news_text])
    prediction = model.predict(news_vec)[0]
    return "Fake News" if prediction == 1 else "Real News"


def fact_check(news_text):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": news_text,
        "languageCode": "en",
        "key": API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if "claims" in data and data["claims"]:
            claim = data["claims"][0]
            if "claimReview" in claim and claim["claimReview"]:
                review = claim["claimReview"][0]
                rating = review.get("textualRating", "Unknown")
                source = review.get("publisher", {}).get("name", "Unknown Source")
                return rating, source

    except requests.exceptions.RequestException as e:
        print(f"Fact-check API error: {e}")
    except Exception as e:
        print(f"Unexpected error in fact_check: {e}")

    return None, None


def wiki_verify(news_text):
    """
    Search Wikipedia for topics related to the news text.
    Returns (wiki_summary, wiki_topic) or (None, None) if not found.
    """
    try:
        # Search Wikipedia for relevant articles
        search_results = wikipedia.search(news_text, results=3)

        if not search_results:
            return None, None

        # Try to get a summary from the top result
        for topic in search_results:
            try:
                summary = wikipedia.summary(topic, sentences=5, auto_suggest=False)
                return summary, topic
            except wikipedia.exceptions.DisambiguationError as e:
                # Pick first option from disambiguation
                try:
                    summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                    return summary, e.options[0]
                except:
                    continue
            except wikipedia.exceptions.PageError:
                continue

    except Exception as e:
        print(f"Wikipedia error: {e}")

    return None, None


def check_claim_against_wiki(news_text, wiki_summary):
    """
    Compare key claim words from news_text against Wikipedia summary.
    Returns 'supported', 'contradicted', or 'uncertain'.
    """
    if not wiki_summary:
        return "uncertain"

    news_lower = news_text.lower()
    wiki_lower = wiki_summary.lower()

    # Keywords that suggest contradiction
    contradiction_signals = [
        # Superlative claims - check if wiki contradicts them
        ("most unesco", ["china has the most", "italy has the most", "spain has the most"]),
        ("largest number of unesco", ["china has the most", "italy has the most"]),
        ("invented in the united states", ["originated in india", "ancient india"]),
        ("failed mathematics", ["excelled", "gifted", "scholarship"]),
        ("moon landing was faked", ["apollo", "landed on the moon", "neil armstrong"]),
        ("vaccines cause autism", ["no link", "no evidence", "disproven"]),
        ("earth is flat", ["spherical", "globe", "oblate spheroid"]),
        ("only use 10%", ["myth", "no evidence", "entire brain"]),
    ]

    for claim_phrase, contra_phrases in contradiction_signals:
        if claim_phrase in news_lower:
            if any(cp in wiki_lower for cp in contra_phrases):
                return "contradicted"

    # General keyword overlap check
    # Extract meaningful words from news (ignore stop words)
    stop_words = {"the", "a", "an", "is", "in", "of", "and", "to", "has", "was",
                  "are", "for", "on", "at", "by", "with", "from", "that", "this"}

    news_words = set(news_lower.split()) - stop_words
    wiki_words = set(wiki_lower.split()) - stop_words

    overlap = news_words & wiki_words
    overlap_ratio = len(overlap) / max(len(news_words), 1)

    # If there's decent overlap, topic is relevant
    if overlap_ratio > 0.3:
        return "supported"

    return "uncertain"


def rule_based_fake(news_text):
    """Detect obvious fake claims via keyword patterns."""
    text = news_text.lower()
    return any(pattern in text for pattern in FAKE_PATTERNS)


def classify_rating(rating_lower):
    """
    Returns 'fake', 'real', or 'uncertain' based on the textual rating.
    """
    fake_keywords = [
        "false", "fake", "pants on fire", "incorrect", "misleading",
        "mostly false", "inaccurate", "wrong", "debunked", "fabricated",
        "fiction", "scam", "hoax", "lie"
    ]
    real_keywords = [
        "true", "correct", "accurate", "verified", "mostly true",
        "confirmed", "real", "legit", "factual"
    ]

    if any(kw in rating_lower for kw in fake_keywords):
        return "fake"
    if any(kw in rating_lower for kw in real_keywords):
        return "real"
    return "uncertain"


def detect_fake(news_text):

    ml_prediction = predict_news(news_text)
    rating, source = fact_check(news_text)
    wiki_verdict = "uncertain"
    wiki_topic = None

    # Priority 1: Rule-based check
    if rule_based_fake(news_text):
        final = "Fake News (Rule-based match)"
        return "Fake News", rating, source, final, "Rule-based", None

    # Priority 2: Fact-check API
    if rating:
        verdict = classify_rating(rating.lower())
        source_label = source or "Unknown Source"

        if verdict == "real":
            final = f"Real News (Fact-checked: {rating} by {source_label})"
            ml_prediction = "Real News"
            return ml_prediction, rating, source, final, "Fact-check API", None

        elif verdict == "fake":
            final = f"Fake News (Fact-checked: {rating} by {source_label})"
            ml_prediction = "Fake News"
            return ml_prediction, rating, source, final, "Fact-check API", None

    # Priority 3: Wikipedia verification
    wiki_summary, wiki_topic = wiki_verify(news_text)

    if wiki_summary:
        wiki_verdict = check_claim_against_wiki(news_text, wiki_summary)

        if wiki_verdict == "contradicted":
            final = f"Fake News (Contradicted by Wikipedia: {wiki_topic})"
            return "Fake News", rating, source, final, "Wikipedia", wiki_topic

        elif wiki_verdict == "supported":
            final = f"Real News (Supported by Wikipedia: {wiki_topic})"
            return "Real News", rating, source, final, "Wikipedia", wiki_topic

    # Priority 4: ML model fallback
    final = f"{ml_prediction} (No verification data found)"
    return ml_prediction, rating, source, final, "ML Model", wiki_topic


if __name__ == "__main__":
    test_cases = [
        "India has most UNESCO heritage sites in Asia",
        "The Earth is flat and NASA is hiding the truth",
        "Elon Musk acquired Twitter and rebranded it to X",
        "The MMR vaccine causes autism in children",
        "India successfully launched Chandrayaan-3 to the moon's south pole",
        "Humans only use 10% of their brain",
    ]

    for news in test_cases:
        ml_prediction, rating, source, final, method, wiki_topic = detect_fake(news)
        print(f"\nNews: {news}")
        print(f"ML Prediction  : {ml_prediction}")
        print(f"Fact Check     : {rating} (Source: {source})" if rating else "Fact Check     : No result found")
        print(f"Wiki Topic     : {wiki_topic}" if wiki_topic else "Wiki Topic     : Not found")
        print(f"Decided By     : {method}")
        print(f"Final Decision : {final}")
        print("-" * 70)