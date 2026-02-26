import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_resource
def train_model():
    fake = pd.read_csv("Fake_small.csv")
    true = pd.read_csv("True_small.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Handle missing values
    data["text"] = data["text"].fillna("")

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    data["text"] = data["text"].apply(clean_text)

    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, vectorizer


model, vectorizer = train_model()

st.title("📰 AI Fake News Detection System")

user_input = st.text_area("Enter news article text:")

if st.button("Analyze"):
    if user_input:
        cleaned_input = user_input.lower()
        cleaned_input = re.sub(r'http\S+', '', cleaned_input)
        cleaned_input = re.sub(r'[^a-zA-Z]', ' ', cleaned_input)
        cleaned_input = re.sub(r'\s+', ' ', cleaned_input)

        vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(vector)
        probability = model.predict_proba(vector)

        label = "Real News" if prediction[0] == 1 else "Fake News"
        confidence = float(max(probability[0]) * 100)

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {round(confidence,2)}%")
    else:
        st.warning("Please enter text.")
