import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ✅ Only ONE clean_text function — no dateline regex
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

# Load dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake['label'] = 0
true['label'] = 1

df = pd.concat([fake, true], ignore_index=True)
df['content'] = df['title'] + " " + df['text']
df = df[['content', 'label']].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text
df['content'] = df['content'].apply(clean_text)

# Split FIRST, then vectorize
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['content'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

# Fit vectorizer ONLY on training data
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Train model
model = SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as confusion_matrix.png")

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("\nModel and vectorizer saved.")

# Sanity tests
print("\nTest 1 (Real):", model.predict(vectorizer.transform(["The United States government announced new economic sanctions against Russia targeting financial institutions"])))
print("Test 2 (Real):", model.predict(vectorizer.transform(["Washington President signed the federal budget bill today after weeks of negotiation in congress"])))
print("Test 3 (Fake):", model.predict(vectorizer.transform(["SHOCKING: Secret elites control the world hidden conspiracy revealed"])))
print("Test 4 (Fake):", model.predict(vectorizer.transform(["government announces new policy"])))