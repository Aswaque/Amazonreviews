import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load Dataset
df = pd.read_csv('reviews.csv')  # File must exist in the same folder

# 2. Data Preprocessing Function
def preprocess(text):
    text = re.sub('<.*?>', '', str(text))  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic chars
    tokens = text.lower().split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_review'] = df['review'].astype(str).apply(preprocess)

# 3. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review']).toarray()

# Encode sentiment labels
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
y = df['sentiment']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training - Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# 6. Model Training - Support Vector Machine (optional)
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# 7. Model Evaluation
def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n--- {model_name} Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

evaluate(lr_model, X_test, y_test, "Logistic Regression")
evaluate(svm_model, X_test, y_test, "Support Vector Machine")

# 8. Save Model and Vectorizer for Deployment
with open('sentiment_lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open('sentiment_svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Models and vectorizer saved successfully.")
