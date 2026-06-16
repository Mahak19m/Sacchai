import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# ─────────────────────────────────────────
# Entity type labels we care about
# ─────────────────────────────────────────
RELEVANT_ENTITIES = {
    "PERSON"  : "👤 Person",
    "ORG"     : "🏢 Organization",
    "GPE"     : "🌍 Location",
    "LOC"     : "📍 Place",
    "EVENT"   : "📅 Event",
    "LAW"     : "⚖️ Law/Policy",
    "NORP"    : "🏳️ Nationality/Group"
}

# ─────────────────────────────────────────
# Main NER function
# ─────────────────────────────────────────
def extract_entities(text):
    """
    Extract named entities from article text.
    Returns categorized entities with counts.
    """

    # spaCy works best on shorter text — use first 5000 chars
    doc = nlp(text[:5000])

    entities = {}

    for ent in doc.ents:
        if ent.label_ not in RELEVANT_ENTITIES:
            continue

        label     = RELEVANT_ENTITIES[ent.label_]
        ent_text  = ent.text.strip()

        # Skip very short or numeric entities
        if len(ent_text) < 2 or ent_text.isdigit():
            continue

        if label not in entities:
            entities[label] = []

        # Avoid duplicates
        if ent_text not in entities[label]:
            entities[label].append(ent_text)

    # No entities found
    if not entities:
        return {
            "status"  : "no_results",
            "message" : "No named entities found in this text.",
            "entities": {}
        }

    return {
        "status"  : "success",
        "message" : f"{sum(len(v) for v in entities.values())} entities found",
        "entities": entities
    }


# ─────────────────────────────────────────
# Test
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_text = """
    Prime Minister Narendra Modi met with US President Joe Biden at the 
    White House in Washington DC on Monday. The United Nations confirmed 
    the meeting was focused on trade relations between India and America.
    The BJP spokesperson said the Indo-US partnership is stronger than ever.
    """

    result = extract_entities(test_text)

    print(f"Status  : {result['status']}")
    print(f"Message : {result['message']}")

    for label, items in result['entities'].items():
        print(f"\n{label}:")
        for item in items:
            print(f"   - {item}")