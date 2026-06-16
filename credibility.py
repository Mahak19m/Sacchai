import re
from utils import predict_news, get_sentiment, get_top_words
from news_checker import check_related_news

# ─────────────────────────────────────────
# Sensational/Clickbait keywords
# ─────────────────────────────────────────
SENSATIONAL_WORDS = [
    'shocking', 'secret', 'conspiracy', 'revealed', 'exposed',
    'breaking', 'urgent', 'alert', 'banned', 'hidden', 'miracle',
    'hoax', 'fraud', 'scam', 'fake', 'lies', 'cover up', 'coverup',
    'illuminati', 'deep state', 'mainstream media', 'they dont want',
    'wake up', 'share before deleted', 'going viral', 'must watch',
    'you wont believe', 'what they are hiding', 'truth about'
]

RELIABLE_WORDS = [
    'according to', 'officials said', 'confirmed', 'reported',
    'announced', 'statement', 'research', 'study', 'data shows',
    'government', 'minister', 'department', 'university', 'institute',
    'percent', 'statistics', 'evidence', 'analysis', 'survey'
]

# ─────────────────────────────────────────
# Individual scoring functions
# ─────────────────────────────────────────

def score_ml_prediction(confidence, result):
    """Score based on ML model prediction and confidence"""
    if result == "Real":
        if confidence >= 80:
            return 25, "✅ ML model strongly predicts Real news"
        elif confidence >= 60:
            return 18, "✅ ML model predicts Real news"
        else:
            return 8, "⚠️ ML model weakly predicts Real news"
    else:
        if confidence >= 80:
            return 0, "❌ ML model strongly predicts Fake news"
        elif confidence >= 60:
            return 3, "❌ ML model predicts Fake news"
        else:
            return 6, "⚠️ ML model weakly predicts Fake news"


def score_sentiment(sentiment_label, sentiment_score):
    """Score based on sentiment — neutral is more credible"""
    if sentiment_label == "Neutral":
        return 20, "✅ Neutral tone detected — more credible"
    elif sentiment_label == "Positive":
        if abs(sentiment_score) < 0.3:
            return 15, "✅ Mildly positive tone — acceptable"
        else:
            return 8, "⚠️ Strongly positive tone — possible bias"
    else:
        if abs(sentiment_score) < 0.3:
            return 15, "✅ Mildly negative tone — acceptable"
        else:
            return 5, "⚠️ Highly negative/emotional tone detected"


def score_sensational_words(text):
    """Penalize sensational/clickbait language"""
    text_lower = text.lower()
    found = [w for w in SENSATIONAL_WORDS if w in text_lower]

    if len(found) == 0:
        return 20, "✅ No sensational keywords found"
    elif len(found) <= 2:
        return 10, f"⚠️ Some sensational words found: {', '.join(found[:2])}"
    else:
        return 0, f"❌ Many sensational keywords found: {', '.join(found[:3])}..."


def score_reliable_words(text):
    """Reward use of reliable/journalistic language"""
    text_lower = text.lower()
    found = [w for w in RELIABLE_WORDS if w in text_lower]

    if len(found) >= 5:
        return 10, f"✅ Strong journalistic language detected"
    elif len(found) >= 2:
        return 7, f"✅ Some reliable language detected"
    else:
        return 3, "⚠️ Limited journalistic language found"


def score_article_length(text):
    """Longer articles tend to be more credible"""
    length = len(text)

    if length >= 3000:
        return 10, "✅ Detailed article — good length"
    elif length >= 1000:
        return 7, "✅ Adequate article length"
    elif length >= 500:
        return 4, "⚠️ Short article — limited context"
    else:
        return 0, "❌ Very short text — insufficient for analysis"

def score_source_verification(text):
    """
    Check if related news exists from trusted sources.
    If zero trusted sources found — big credibility penalty.
    """
    try:
        news = check_related_news(text[:200])
        
        trusted = news.get("trusted_count", 0)
        total   = news.get("total_count", 0)

        if news["status"] != "success" or total == 0:
            return 5, "⚠️ No related news found — unverifiable claim"

        if trusted >= 3:
            return 15, f"✅ {trusted}/{total} related articles from trusted sources"
        elif trusted >= 1:
            return 8, f"⚠️ Only {trusted}/{total} related articles from trusted sources"
        else:
            return 0, f"❌ No trusted sources found for this claim — high risk"

    except:
        return 5, "⚠️ Source verification unavailable"

def score_named_sources(text):
    """
    Real news names sources. Fake news uses vague attribution.
    Penalize vague source language heavily.
    """
    text_lower = text.lower()

    # Vague source patterns common in fake news
    VAGUE_SOURCES = [
        'officials say', 'officials are refusing',
        'sources say', 'some people say',
        'many people', 'hundreds of witnesses',
        'experts say', 'scientists say',
        'people are saying', 'everyone knows',
        'it is reported', 'it is claimed',
        'anonymous sources', 'inside sources',
        'someone said', 'they said',
        'witnesses say', 'locals say'
    ]

    # Strong named source patterns
    NAMED_SOURCES = [
        'according to', 'said in a statement',
        'told reporters', 'said on',
        'ministry of', 'department of',
        'prime minister', 'president',
        'dr.', 'prof.', 'professor',
        'university of', 'institute of',
        'percent', 'survey of'
    ]

    vague_found = [w for w in VAGUE_SOURCES if w in text_lower]
    named_found = [w for w in NAMED_SOURCES if w in text_lower]

    if len(vague_found) >= 2 and len(named_found) == 0:
        return 0, f"❌ Multiple vague sources, no named attribution — high fake risk"
    elif len(vague_found) >= 1 and len(named_found) == 0:
        return 3, f"⚠️ Vague source language detected: '{vague_found[0]}'"
    elif len(named_found) >= 3:
        return 10, f"✅ Strong named source attribution detected"
    elif len(named_found) >= 1:
        return 7, f"✅ Some named sources detected"
    else:
        return 5, "⚠️ Limited source attribution found"

# ─────────────────────────────────────────
# Main credibility scoring function
# ─────────────────────────────────────────

def get_credibility_score(text):
    """
    Returns full credibility analysis
    Total score: 100 points
    - ML Prediction:      40 points
    - Sentiment:          20 points
    - Sensational words:  20 points
    - Reliable language:  10 points
    - Article Length:     10 points
    """

    # Gate: too short for reliable analysis
    if len(text.split()) < 30:
        return {
            "total_score"   : 0,
            "label"         : "Insufficient Text",
            "emoji"         : "⚠️",
            "ml_prediction" : "Unknown",
            "confidence"    : 0,
            "sentiment"     : "Unknown",
            "breakdown"     : {},
            "error"         : "Text too short for reliable analysis. Please provide at least 30 words for accurate results."
        }

    # Get base predictions
    result, confidence = predict_news(text)
    sentiment_label, sentiment_score = get_sentiment(text)

    # Individual scores
    ml_score,   ml_reason   = score_ml_prediction(confidence, result)
    sent_score, sent_reason = score_sentiment(sentiment_label, sentiment_score)
    sens_score, sens_reason = score_sensational_words(text)
    rel_score,  rel_reason  = score_reliable_words(text)
    len_score,  len_reason  = score_article_length(text)
    src_score,  src_reason  = score_source_verification(text)
    nam_score,  nam_reason  = score_named_sources(text)

    # Total score — now out of 125, normalized to 100
    # Total score — now out of 110, normalized to 100
    raw_total = ml_score + sent_score + sens_score + rel_score + len_score + src_score + nam_score
    total = min(100, int((raw_total / 110) * 100))

    # Hard cap — if named sources AND source verification both fail, cap at 40
    if nam_score == 0 and src_score <= 5:
        total = min(total, 40)

    # Hard cap — if named sources fail completely, cap at 50
    if nam_score == 0:
        total = min(total, 50)

    # Credibility label
    if total >= 75:
        label = "Highly Credible"
        emoji = "🟢"
    elif total >= 55:
        label = "Likely Credible"
        emoji = "🟡"
    elif total >= 35:
        label = "Questionable"
        emoji = "🟠"
    else:
        label = "Not Credible"
        emoji = "🔴"

    return {
        "total_score"    : total,
        "label"          : label,
        "emoji"          : emoji,
        "ml_prediction"  : result,
        "confidence"     : confidence,
        "sentiment"      : sentiment_label,
       "breakdown"      : {
            "ML Prediction"      : (ml_score,   40, ml_reason),
            "Sentiment"          : (sent_score, 20, sent_reason),
            "Sensational Words"  : (sens_score, 20, sens_reason),
            "Reliable Language"  : (rel_score,  10, rel_reason),
            "Article Length"     : (len_score,  10, len_reason),
            "Source Verification": (src_score,  15, src_reason),
            "Named Sources"      : (nam_score,  10, nam_reason),
        }
    }


# ─────────────────────────────────────────
# Test
# ─────────────────────────────────────────
# if __name__ == "__main__":

#     test_cases = [
#         {
#             "label": "Real News",
#             "text": """The United States Federal Reserve raised interest rates by 
#             0.25 percent on Wednesday according to officials. The treasury department 
#             confirmed the decision following months of data analysis and research. 
#             Government ministers announced the policy would take effect immediately. 
#             Statistics show inflation has declined by 2 percent over the past quarter 
#             according to the latest survey conducted by the institute of economic research."""
#         },
#         {
#             "label": "Fake News",
#             "text": """SHOCKING secret revealed! The deep state has been hiding the 
#             truth about the miracle cure they dont want you to know. Share before deleted! 
#             Exposed: The illuminati conspiracy cover up that mainstream media wont tell you. 
#             Wake up people! You wont believe what they are hiding from us all!"""
#         }
#     ]

#     for case in test_cases:
#         print("\n" + "=" * 55)
#         print(f"Testing: {case['label']}")
#         print("=" * 55)

#         result = get_credibility_score(case['text'])

#         print(f"\n{result['emoji']} Credibility: {result['label']}")
#         print(f"   Total Score : {result['total_score']}/100")
#         print(f"   Prediction  : {result['ml_prediction']} ({result['confidence']}%)")
#         print(f"   Sentiment   : {result['sentiment']}")
#         print(f"\n📊 Score Breakdown:")
#         for category, (score, max_score, reason) in result['breakdown'].items():
#             print(f"   {reason} [{score}/{max_score}]")



if __name__ == "__main__":
    from url_extractor import extract_article_from_url

    url = "https://www.ndtv.com/opinion/us-israel-iran-war-why-indias-partners-must-always-know-that-it-has-other-options-too-11365151?pfrom=home-ndtv_topstories"
    
    title, text, error = extract_article_from_url(url)
    
    if error:
        print("Error:", error)
    else:
        full_content = f"{title} {text}"
        print(f"Article: {title[:60]}")
        print(f"Length: {len(full_content)} characters\n")
        
        result = get_credibility_score(full_content)
        
        print(f"{result['emoji']} Credibility: {result['label']}")
        print(f"   Total Score : {result['total_score']}/100")
        print(f"   Prediction  : {result['ml_prediction']} ({result['confidence']}%)")
        print(f"   Sentiment   : {result['sentiment']}")
        print(f"\n📊 Score Breakdown:")
        for category, (score, max_score, reason) in result['breakdown'].items():
            print(f"   {reason} [{score}/{max_score}]")