import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK Downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
with open('sentiment_lr_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load amazonreviews.csv from specific path
csv_path = r"C:\Users\aswaq\Documents\Sentiment_project\amazonreviews.csv"
df = pd.read_csv(csv_path)

# Preprocessing function
def preprocess(text):
    text = re.sub('<.*?>', '', str(text))
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🛒 Amazon Product Review Sentiment Analyzer")
st.markdown("Select a product to analyze customer review sentiments and decide if it's a **Good Product** or not.")

# Select product
product_names = df['product'].unique()
selected_product = st.selectbox("Select a product:", sorted(product_names))

if selected_product:
    product_reviews = df[df['product'] == selected_product].copy()
    
    # Preprocess reviews
    product_reviews['cleaned_review'] = product_reviews['review'].apply(preprocess)
    
    # Vectorize and predict
    X = vectorizer.transform(product_reviews['cleaned_review']).toarray()
    product_reviews['predicted_sentiment'] = model.predict(X)
    
    # Count sentiments
    pos_count = (product_reviews['predicted_sentiment'] == 1).sum()
    neg_count = (product_reviews['predicted_sentiment'] == 0).sum()
    total = len(product_reviews)

    # Show pie chart
    st.subheader("📊 Sentiment Breakdown")
    fig, ax = plt.subplots()
    ax.pie([pos_count, neg_count], labels=['Positive', 'Negative'], colors=['green', 'red'], autopct='%1.1f%%')
    st.pyplot(fig)

    # Overall recommendation
    if pos_count / total >= 0.6:
        st.success(f"✅ **{selected_product}** is a Good Product!")
    else:
        st.error(f"❌ **{selected_product}** is Not Recommended based on review sentiment.")

    # Sample reviews
    st.subheader("🔍 Sample Customer Reviews")
    sample_reviews = product_reviews[['review', 'predicted_sentiment']].replace({1: 'Positive 😊', 0: 'Negative 😞'})
    st.dataframe(sample_reviews.sample(5))
