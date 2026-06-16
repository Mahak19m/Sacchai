import pickle
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

# Download stopwords quietly
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ Same clean_text as model.py (must be identical!)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\(reuters\)', '', text)
    text = re.sub(r'reuters', '', text)
    # ✅ Removed the aggressive dateline regex — it was hurting predictions
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ✅ Predict using decision_function (works with SGDClassifier)
def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    # decision_function gives raw confidence score
    decision = model.decision_function(vector)[0]

    # Convert to 0-1 range using sigmoid
    import math
    confidence = 1 / (1 + math.exp(-abs(decision)))

    label = "Fake" if prediction == 0 else "Real"

    return label, round(confidence * 100, 2)  # return as percentage

# Sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity

    if score > 0.1:
        return "Positive", round(score, 2)
    elif score < -0.1:
        return "Negative", round(score, 2)
    else:
        return "Neutral", round(score, 2)

# ✅ Top words with their actual TF-IDF scores
def get_top_words(text, n=5):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()
    scores = vector.toarray()[0]

    top_indices = scores.argsort()[-n:][::-1]
    top_words = [(feature_names[i], round(scores[i], 4)) for i in top_indices]

    # Filter out zero-score words
    top_words = [(word, score) for word, score in top_words if score > 0]

    return top_words



if __name__ == "__main__":
    test_text = """The United States government announced new economic sanctions 
    against Russia on Monday, targeting key financial institutions 
    and energy companies. The Treasury Department said the measures 
    were in response to continued aggression in Eastern Europe. 
    Senior officials stated that the sanctions would take effect 
    immediately and affect over 50 entities linked to the Kremlin."""
    
    print("=== DEBUG ===")
    print("Cleaned text:", clean_text(test_text))
    print("Prediction:", predict_news(test_text))
    print("Top words:", get_top_words(test_text))