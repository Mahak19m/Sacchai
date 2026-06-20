# Sachhai — Fake News Detection System

> *Sachhai (सच्चाई) means "truth" in Hindi.*

A credibility assessment system for English and Hindi news articles and WhatsApp forwards. Unlike binary classifiers that just label something as "fake" or "real", Sachhai generates a **credibility score out of 100** using a 6-layer analysis pipeline.

---

## Features

- **6-layer credibility scoring** — combines ML classification, fact-checking, NER, sentiment, source analysis, and OCR
- **Multilingual support** — works on both English and Hindi content
- **Multiple input modes** — paste text, enter a URL, or upload a WhatsApp screenshot image
- **99.83% accuracy** on ISOT Fake News Dataset (44,898 articles)
- **Interactive Streamlit web app** — instant results with score breakdown

---

## How It Works

| Layer | Module | What it does |
|-------|--------|--------------|
| 1 | `model.py` | SGDClassifier with TF-IDF vectorisation (trained on ISOT dataset) |
| 2 | `fact_check.py` | Live Google Fact Check API lookup |
| 3 | `ner_extractor.py` | Named Entity Recognition via spaCy |
| 4 | `credibility.py` | Source and domain credibility analysis |
| 5 | `url_extractor.py` | URL metadata and link analysis |
| 6 | `ocr_extractor.py` | Tesseract OCR for WhatsApp screenshot input |

Final score (0–100) is computed by weighting outputs from all 6 layers.

---

## Tech Stack

- **ML / NLP:** scikit-learn, spaCy, TextBlob, NLTK
- **OCR:** Tesseract
- **APIs:** Google Fact Check API
- **Web App:** Streamlit
- **Data:** ISOT Fake News Dataset (Kaggle)
- **Language:** Python

---

## Project Structure

```
Sachhai/
├── app.py               # Streamlit web app entry point
├── model.py             # ML model training and inference
├── credibility.py       # Credibility scoring engine
├── fact_check.py        # Google Fact Check API integration
├── ner_extractor.py     # Named entity recognition
├── ocr_extractor.py     # OCR for image input
├── url_extractor.py     # URL analysis
├── news_checker.py      # Core pipeline orchestrator
├── utils.py             # Helper functions
├── test.py              # Test suite
├── requirements.txt     # Dependencies
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Mahak19m/Sachhai.git
cd Sachhai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set up Tesseract OCR
- **Windows:** Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux:** `sudo apt install tesseract-ocr`
- **Mac:** `brew install tesseract`

### 4. Add your Google Fact Check API key
Create a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| Dataset | ISOT Fake News Dataset |
| Training samples | 44,898 articles |
| Algorithm | SGDClassifier + TF-IDF |
| Test Accuracy | **99.83%** |

---

## About

Built by **Mahak** — B.Tech student in Computer Science / Data Analytics at Dr. Ambedkar Institute of Technology for Handicapped, Kanpur.

Part of the [AI Careers for Women Program](https://edunetfoundation.org/) — Microsoft · Edunet Foundation · SAP India · Govt. of India 2025–26.
