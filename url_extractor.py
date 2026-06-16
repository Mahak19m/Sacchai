import requests
from newspaper import Article
import urllib.request

def extract_article_from_url(url):
    """
    Extract title and text from a news article URL
    with multiple fallback methods
    """
    # Method 1 — newspaper3k with browser headers
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        article = Article(url)
        article.set_html(response.text)
        article.parse()

        if article.text and len(article.text) > 100:
            title = article.title if article.title else "No title found"
            return title, article.text, None

    except Exception as e:
        pass  # Try next method

    # Method 2 — direct newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()

        if article.text and len(article.text) > 100:
            title = article.title if article.title else "No title found"
            return title, article.text, None

    except Exception as e:
        pass  # Try next method

    # Method 3 — requests only
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        article = Article(url)
        article.set_html(response.text)
        article.parse()

        if article.text and len(article.text) > 100:
            title = article.title if article.title else "No title found"
            return title, article.text, None

    except Exception as e:
        return None, None, f"Could not extract article. Site may be blocking scrapers."


def get_full_content(title, text):
    return f"{title} {text}"


# ─────────────────────────────────────────
# Test with multiple URLs
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "https://www.reuters.com/world/us/",
        "https://timesofindia.indiatimes.com/world",
        "https://www.ndtv.com/world-news",
    ]

    for url in test_urls:
        print(f"\nTesting: {url}")
        title, text, error = extract_article_from_url(url)
        if error:
            print(f"❌ Error: {error}")
        else:
            print(f"✅ Title: {title}")
            print(f"✅ Text length: {len(text)} characters")
            print(f"✅ Preview: {text[:150]}...")
        print("-" * 60)