import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ✅ Same clean_text as model.py
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\(reuters\)', '', text)
    text = re.sub(r'reuters', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ─────────────────────────────────────────
# 1. Load model and vectorizer
# ─────────────────────────────────────────
print("=" * 50)
print("        FAKE NEWS DETECTION - TEST REPORT")
print("=" * 50)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
print("\n✅ Model and vectorizer loaded successfully")
print(f"   Model type: {type(model).__name__}")

# ─────────────────────────────────────────
# 2. Dataset Statistics
# ─────────────────────────────────────────
print("\n" + "─" * 50)
print("📊 DATASET STATISTICS")
print("─" * 50)

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake['label'] = 0
true['label'] = 1

df = pd.concat([fake, true], ignore_index=True)
df['content'] = df['title'] + " " + df['text']
df = df[['content', 'label']].dropna()

print(f"   Total articles     : {len(df)}")
print(f"   Fake articles      : {len(fake)}")
print(f"   Real articles      : {len(true)}")
print(f"   Missing values     : {df.isnull().sum().sum()}")
print(f"   Avg article length : {df['content'].apply(len).mean():.0f} characters")
print(f"   Max article length : {df['content'].apply(len).max()} characters")
print(f"   Min article length : {df['content'].apply(len).min()} characters")

# ─────────────────────────────────────────
# 3. Label Distribution Plot
# ─────────────────────────────────────────
plt.figure(figsize=(5, 3))
df['label'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'], edgecolor='black')
plt.xticks([0, 1], ['Fake', 'Real'], rotation=0)
plt.title('Dataset Label Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('label_distribution.png')
plt.close()
print("\n✅ Label distribution plot saved as label_distribution.png")

# ─────────────────────────────────────────
# 4. Model Evaluation on Test Set
# ─────────────────────────────────────────
print("\n" + "─" * 50)
print("🤖 MODEL EVALUATION")
print("─" * 50)

from sklearn.model_selection import train_test_split
df['content'] = df['content'].apply(clean_text)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['content'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

X_test = vectorizer.transform(X_test_raw)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Accuracy  : {accuracy * 100:.2f}%")
print(f"   Test size : {len(y_test)} articles")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# ─────────────────────────────────────────
# 5. Confusion Matrix
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_test.png')
plt.close()
print("✅ Confusion matrix saved as confusion_matrix_test.png")

# ─────────────────────────────────────────
# 6. Custom Prediction Tests
# ─────────────────────────────────────────
print("\n" + "─" * 50)
print("🧪 CUSTOM PREDICTION TESTS")
print("─" * 50)

test_cases = [
    ("The Senate passed a new healthcare bill with bipartisan support after months of negotiations.", "Real"),
    ("NASA announces successful launch of new climate monitoring satellite into orbit.", "Real"),
    ("BREAKING: Scientists discover cure for all diseases hidden by government!", "Fake"),
    ("Shocking truth revealed: Moon landing was staged in Hollywood studio!", "Fake"),
    ("Federal Reserve raises interest rates by 0.25 percent amid inflation concerns.", "Real"),
    ("Secret society controls world economy through shadow banking system!", "Fake"),
]

print(f"\n{'News Text':<65} {'Expected':<10} {'Got':<10} {'Result'}")
print("-" * 100)

correct = 0
for text, expected in test_cases:
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    got = "Real" if prediction == 1 else "Fake"
    status = "✅" if got == expected else "❌"
    if got == expected:
        correct += 1
    print(f"{text[:63]:<65} {expected:<10} {got:<10} {status}")

print(f"\n   Custom Test Accuracy: {correct}/{len(test_cases)} correct")

# ─────────────────────────────────────────
# 7. Summary
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("📋 SUMMARY")
print("=" * 50)
print(f"   Dataset Size      : {len(df)} articles")
print(f"   Model             : SGDClassifier (PA-I)")
print(f"   Vectorizer        : TF-IDF (10000 features, bigrams)")
print(f"   Test Accuracy     : {accuracy * 100:.2f}%")
print(f"   Custom Tests      : {correct}/{len(test_cases)} passed")
print("=" * 50)
print("\n✅ All tests completed successfully!")