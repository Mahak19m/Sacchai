import feedparser
import requests
from urllib.parse import quote

# ─────────────────────────────────────────
# Trusted news domains
# ─────────────────────────────────────────
TRUSTED_DOMAINS = [
    # International
    'reuters.com', 'bbc.com', 'bbc.co.uk', 'apnews.com',
    'theguardian.com', 'aljazeera.com',
    # Indian mainstream
    'ndtv.com', 'thehindu.com', 'hindustantimes.com',
    'indianexpress.com', 'timesofindia.com', 'scroll.in',
    'thewire.in', 'livemint.com', 'businessstandard.com',
    'telegraphindia.com', 'deccanherald.com',
    # Indian government/official
    'pib.gov.in', 'india.gov.in', 'mygov.in',
    'investindia.gov.in'
]

# ─────────────────────────────────────────
# Extract short query from text
# ─────────────────────────────────────────
def extract_query(text, max_words=8):
    words = text.strip().split()
    return ' '.join(words[:max_words])


# ─────────────────────────────────────────
# Check if a URL is from a trusted domain
# ─────────────────────────────────────────
def is_trusted_source(url):
    # Google News RSS wraps URLs — check the source title instead
    url_lower = url.lower()
    return any(domain in url_lower for domain in TRUSTED_DOMAINS)

def is_trusted_source_by_name(source_name):
    """Check source name against trusted list"""
    TRUSTED_NAMES = [
        'reuters', 'bbc', 'ndtv', 'the hindu', 'hindustan times',
        'indian express', 'ap news', 'associated press', 'the guardian',
        'al jazeera', 'times of india', 'scroll', 'the wire', 'mint',
        'live mint', 'business standard', 'deccan herald',
        'telegraph india', 'pib', 'press information bureau',
        'invest india', 'india today', 'the print', 'quint'
    ]
    source_lower = source_name.lower()
    return any(name in source_lower for name in TRUSTED_NAMES)


# ─────────────────────────────────────────
# Main function
# ─────────────────────────────────────────
def check_related_news(text):
    """
    Search Google News RSS for related articles.
    Returns related articles with source credibility info.
    """

    query = extract_query(text)
    encoded_query = quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(rss_url)

        if not feed.entries:
            return {
                "status"         : "no_results",
                "message"        : "No related news articles found.",
                "articles"       : [],
                "trusted_count"  : 0,
                "total_count"    : 0
            }

        articles = []
        for entry in feed.entries[:5]:  # Max 5 results
            url        = entry.get("link", "")
            title      = entry.get("title", "No title")
            source     = entry.get("source", {}).get("title", "Unknown")
            published  = entry.get("published", "N/A")[:16]
            trusted    = is_trusted_source(url) or is_trusted_source_by_name(source)

            articles.append({
                "title"    : title,
                "source"   : source,
                "url"      : url,
                "published": published,
                "trusted"  : trusted
            })

        trusted_count = sum(1 for a in articles if a["trusted"])

        return {
            "status"        : "success",
            "message"       : f"{len(articles)} related articles found",
            "articles"      : articles,
            "trusted_count" : trusted_count,
            "total_count"   : len(articles)
        }

    except Exception as e:
        return {
            "status"        : "error",
            "message"       : f"News search failed: {str(e)}",
            "articles"      : [],
            "trusted_count" : 0,
            "total_count"   : 0
        }


# ─────────────────────────────────────────
# Test
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_text = "India launches new economic policy to boost manufacturing sector"
    result = check_related_news(test_text)

    print(f"Status  : {result['status']}")
    print(f"Message : {result['message']}")
    print(f"Trusted : {result['trusted_count']}/{result['total_count']}")

    for i, a in enumerate(result['articles'], 1):
        trust_tag = "✅ Trusted" if a['trusted'] else "⚠️ Unknown"
        print(f"\n--- Article {i} ---")
        print(f"Title   : {a['title'][:70]}")
        print(f"Source  : {a['source']}")
        print(f"Trust   : {trust_tag}")
        print(f"Date    : {a['published']}")