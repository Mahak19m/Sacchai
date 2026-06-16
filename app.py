import streamlit as st
from utils import predict_news, get_sentiment, get_top_words
from url_extractor import extract_article_from_url
from credibility import get_credibility_score
from fact_check import check_facts
from news_checker import check_related_news
from ner_extractor import extract_entities
from ocr_extractor import extract_text_from_image

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Analyzer Pro",
    page_icon="📰",
    layout="wide"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .score-number {
        font-size: 60px;
        font-weight: bold;
        color: white;
    }
    .score-label {
        font-size: 22px;
        color: white;
        margin-top: 5px;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px 0;
    }
    .breakdown-item {
        background: white;
        padding: 12px 18px;
        border-radius: 8px;
        margin: 6px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        font-size: 15px;
    }
    .keyword-badge {
        background-color: #e8f4f8;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 4px;
        display: inline-block;
        font-size: 13px;
        color: #2c3e50;
        border: 1px solid #bee3f8;
    }
    .stTextArea textarea {
        font-size: 15px;
        border-radius: 10px;
    }
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>📰 Fake News Analyzer Pro</h1>
    <p style="font-size:16px; color:#a0aec0;">
        Advanced ML-powered fake news detection with credibility scoring
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    input_mode = st.radio(
        "Select Input Mode:",
        ["✍️ Manual Text", "🔗 URL Input", "📱 WhatsApp Forward"],
        index=0
    )
    st.divider()
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Model:** SGDClassifier (PA-I)  
    **Dataset:** ISOT Fake News  
    **Articles:** 44,898  
    **Accuracy:** 99.83%  
    """)
    st.divider()
    st.markdown("## 📊 Score Guide")
    st.markdown("""
    🟢 **75-100** → Highly Credible  
    🟡 **55-74**  → Likely Credible  
    🟠 **35-54**  → Questionable  
    🔴 **0-34**   → Not Credible  
    """)
    st.divider()
    st.caption("⚠️ For educational purposes only. Always verify from trusted sources.")

# ─────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────
st.markdown("### 📥 Input")

user_text = ""
article_title = ""

if input_mode == "✍️ Manual Text":
    user_text = st.text_area(
        label="",
        placeholder="Paste a news article or headline here...",
        height=200
    )

elif input_mode == "🔗 URL Input":
    url_input = st.text_input(
        label="",
        placeholder="https://www.ndtv.com/your-article-url"
    )
    if url_input:
        with st.spinner("🔍 Extracting article from URL..."):
            title, text, error = extract_article_from_url(url_input)
        if error:
            st.error(f"❌ {error} — Please paste the article text manually instead.")
        else:
            article_title = title
            user_text = f"{title} {text}"
            st.success(f"✅ Article extracted: **{title[:80]}...**")
            with st.expander("📄 View extracted text"):
                st.write(text[:1000] + "...")

elif input_mode == "📱 WhatsApp Forward":
    st.markdown("#### 📱 Upload WhatsApp Forward Screenshot")
    st.caption("Upload a screenshot of a WhatsApp message or forward to analyze it.")

    uploaded_image = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg"],
        help="Take a screenshot of the WhatsApp forward and upload it here"
    )

    if uploaded_image is not None:
        # Show the uploaded image
        st.image(uploaded_image, caption="Uploaded Screenshot", width=300)

        with st.spinner("🔍 Extracting text from image..."):
            ocr_result = extract_text_from_image(uploaded_image)

        if ocr_result["status"] == "success":
            user_text = ocr_result["text"]
            st.success(f"✅ {ocr_result['message']}")
            with st.expander("📄 View extracted text"):
                st.write(user_text)
        else:
            st.error(f"❌ {ocr_result['message']}")


# ─────────────────────────────────────────
# Analyze Button
# ─────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Article", use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# Analysis Results
# ─────────────────────────────────────────
if analyze_btn:
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text or a valid URL before analyzing.")
    else:
        with st.spinner("🤖 Analyzing article..."):
            # Get all results
            credibility    = get_credibility_score(user_text)
            keywords       = get_top_words(user_text)
            fact_results   = check_facts(user_text)
            news_results   = check_related_news(user_text)
            ner_results    = extract_entities(user_text)

        # ── Credibility Score Card ──
        st.markdown("### 📊 Credibility Analysis")

        score = credibility['total_score']
        label = credibility['label']
        emoji = credibility['emoji']

        # Handle insufficient text
        if label == "Insufficient Text":
            st.warning("⚠️ " + credibility['error'])
            st.stop()

        # Score color
        if score >= 75:
            color = "#38a169"
        elif score >= 55:
            color = "#d69e2e"
        elif score >= 35:
            color = "#dd6b20"
        else:
            color = "#e53e3e"

        st.markdown(f"""
        <div class="score-card">
            <div class="score-number">{score}/100</div>
            <div class="score-label">{emoji} {label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metrics Row ──
        m1, m2, m3 = st.columns(3)
        with m1:
            pred = credibility['ml_prediction']
            conf = credibility['confidence']
            st.metric("🧾 Prediction", pred, f"{conf}% confidence")
        with m2:
            sent = credibility['sentiment']
            st.metric("💬 Sentiment", sent)
        with m3:
            st.metric("📏 Article Length", f"{len(user_text)} chars")

        st.divider()

        # ── Score Breakdown ──
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🔍 Score Breakdown")
            for category, (score_val, max_score, reason) in credibility['breakdown'].items():
                pct = int((score_val / max_score) * 100)
                st.markdown(f"""
                <div class="breakdown-item">
                    {reason} <b>[{score_val}/{max_score}]</b>
                </div>
                """, unsafe_allow_html=True)
                st.progress(pct / 100)

        with col_right:
            st.markdown("### 🧠 Key Terms Detected")
            if keywords:
                keyword_html = ""
                for word, score_val in keywords:
                    keyword_html += f'<span class="keyword-badge">🔑 {word} ({score_val})</span> '
                st.markdown(keyword_html, unsafe_allow_html=True)
            else:
                st.info("No significant keywords found.")

            st.divider()

            # ── Interpretation ──
            st.markdown("### 💡 Interpretation")
            pred = credibility['ml_prediction']
            sent = credibility['sentiment']
            total = credibility['total_score']

            if total >= 75:
                st.success("✅ This article shows strong signs of credibility. Language is factual and tone is balanced.")
            elif total >= 55:
                st.info("ℹ️ This article appears mostly credible but has some flags worth noting.")
            elif total >= 35:
                st.warning("⚠️ This article has several credibility concerns. Verify from other sources.")
            else:
                st.error("🚨 This article shows strong signs of being fake or misleading. Do not share without verification.")

            if sent == "Negative" and pred == "Fake":
                st.warning("⚠️ Highly emotional negative language combined with fake prediction is a strong indicator of misinformation.")
            elif sent == "Positive" and pred == "Fake":
                st.warning("⚠️ Unusually positive framing with fake prediction may indicate clickbait or propaganda.")

        st.divider()

                # ── Fact Check Results ──

        # ── Fact Check Results ──
        st.markdown("### 🔎 Fact Check Results")

        if fact_results["status"] == "success":
            st.success(f"✅ {fact_results['message']} from independent fact checkers")
            for i, r in enumerate(fact_results["results"], 1):
                with st.expander(f"📋 Result {i} — {r['source']} ({r['date']})"):
                    st.markdown(f"**Claim:** {r['claim']}")
                    st.markdown(f"**Claimant:** {r['claimant']}")

                    # Color code the verdict
                    verdict_lower = r['verdict'].lower()
                    if any(w in verdict_lower for w in ['false', 'fake', 'misleading', 'incorrect']):
                        st.error(f"❌ Verdict: {r['verdict']}")
                    elif any(w in verdict_lower for w in ['true', 'correct', 'accurate']):
                        st.success(f"✅ Verdict: {r['verdict']}")
                    else:
                        st.warning(f"⚠️ Verdict: {r['verdict']}")

                    if r['url']:
                        st.markdown(f"🔗 [Read full fact check]({r['url']})")

        elif fact_results["status"] == "no_results":
            st.info("ℹ️ No matching fact checks found for this content in our database.")

        else:
            st.warning(f"⚠️ Fact Check unavailable: {fact_results['message']}")

        st.divider()

        # ── Related News ──
        st.markdown("### 📡 Related News Articles")

        if news_results["status"] == "success":
            trusted = news_results["trusted_count"]
            total   = news_results["total_count"]

            # Trusted source ratio bar
            if trusted >= 3:
                st.success(f"✅ {trusted}/{total} related articles from trusted sources")
            elif trusted >= 1:
                st.warning(f"⚠️ {trusted}/{total} related articles from trusted sources")
            else:
                st.error(f"❌ {trusted}/{total} related articles from trusted sources — topic may be obscure or unverified")

            for a in news_results["articles"]:
                trust_icon = "✅" if a["trusted"] else "⚠️"
                with st.expander(f"{trust_icon} {a['source']} — {a['published']}"):
                    st.markdown(f"**{a['title']}**")
                    st.markdown(f"🔗 [Read article]({a['url']})")

        elif news_results["status"] == "no_results":
            st.info("ℹ️ No related news articles found for this content.")

        else:
            st.warning(f"⚠️ News search unavailable: {news_results['message']}")

        st.divider()

        # ── NER Entities ──
        st.markdown("### 🧩 Named Entities Detected")

        if ner_results["status"] == "success":
            st.caption(f"ℹ️ {ner_results['message']} — real news typically references verifiable people, places and organizations")

            for label, items in ner_results["entities"].items():
                st.markdown(f"**{label}**")
                badge_html = ""
                for item in items:
                    badge_html += f'<span class="keyword-badge">{item}</span> '
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown("")
        else:
            st.info("ℹ️ No named entities found — vague or generic content detected.")

        st.divider()

        # ── Related News ──
        st.markdown("### 📡 Related News Articles")