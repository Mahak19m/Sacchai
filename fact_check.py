import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# ─────────────────────────────────────────
# Extract a short searchable query from text
# ─────────────────────────────────────────
def extract_query(text, max_words=10):
    """Take first 10 meaningful words as search query"""
    words = text.strip().split()
    return ' '.join(words[:max_words])


# ─────────────────────────────────────────
# Main fact check function
# ─────────────────────────────────────────
def check_facts(text):
    """
    Query Google Fact Check API with article text.
    Returns list of fact check results or empty list.
    """

    # If no API key configured
    if not API_KEY:
        return {
            "status"  : "unavailable",
            "message" : "Fact Check API key not configured.",
            "results" : []
        }

    query = extract_query(text)

    try:
        response = requests.get(
            BASE_URL,
            params={
                "query"        : query,
                "key"          : API_KEY,
                "languageCode" : "en-US"
            },
            timeout=5  # Don't hang if API is slow
        )

        # Bad API key or quota exceeded
        if response.status_code == 400:
            return {
                "status"  : "error",
                "message" : "Invalid API key. Check your .env file.",
                "results" : []
            }

        if response.status_code == 403:
            return {
                "status"  : "error",
                "message" : "API quota exceeded or key not authorized.",
                "results" : []
            }

        data = response.json()

        # No claims found for this text
        if "claims" not in data or len(data["claims"]) == 0:
            return {
                "status"  : "no_results",
                "message" : "No fact checks found for this content.",
                "results" : []
            }

        # Parse results
        results = []
        for claim in data["claims"][:3]:  # Max 3 results
            
            # Extract claim review safely
            review = claim.get("claimReview", [{}])[0]

            results.append({
                "claim"      : claim.get("text", "N/A"),
                "claimant"   : claim.get("claimant", "Unknown"),
                "verdict"    : review.get("textualRating", "No verdict"),
                "source"     : review.get("publisher", {}).get("name", "Unknown"),
                "url"        : review.get("url", ""),
                "date"       : review.get("reviewDate", "")[:10]
                               if review.get("reviewDate") else "N/A"
            })

        return {
            "status"  : "success",
            "message" : f"{len(results)} fact check(s) found",
            "results" : results
        }

    except requests.exceptions.Timeout:
        return {
            "status"  : "error",
            "message" : "Fact Check API timed out. Skipping.",
            "results" : []
        }

    except requests.exceptions.ConnectionError:
        return {
            "status"  : "error",
            "message" : "No internet connection.",
            "results" : []
        }

    except Exception as e:
        return {
            "status"  : "error",
            "message" : f"Unexpected error: {str(e)}",
            "results" : []
        }


# ─────────────────────────────────────────
# Test
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_text = "Trump claims the election was stolen and voting machines were rigged"
    result = check_facts(test_text)
    
    print(f"Status  : {result['status']}")
    print(f"Message : {result['message']}")
    
    for i, r in enumerate(result['results'], 1):
        print(f"\n--- Result {i} ---")
        print(f"Claim    : {r['claim'][:80]}")
        print(f"Claimant : {r['claimant']}")
        print(f"Verdict  : {r['verdict']}")
        print(f"Source   : {r['source']}")
        print(f"Date     : {r['date']}")
        print(f"URL      : {r['url']}")