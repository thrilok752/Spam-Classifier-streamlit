import streamlit as st
import pickle
import re

# Load saved model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

st.title("📩 Spam Message Classifier")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.error("🚨 This message is Spam")
    else:
        st.success("✅ This message is Not Spam")