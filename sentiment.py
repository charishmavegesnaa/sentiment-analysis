import streamlit as st
import pickle
import numpy as np

# Set page title
st.title("Sentiment Analysis Prediction App")

# Load the trained model and vectorizer
try:
    with open('sentiment_logreg_model.pkl', 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        vectorizer = data['vectorizer']
except FileNotFoundError:
    st.error("Model file 'sentiment_logreg_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Text input for user
text_input = st.text_area("Enter text to analyze sentiment:", "")

# Button to make prediction
if st.button("Predict Sentiment"):
    if text_input.strip():
        # Transform input text
        X_input = vectorizer.transform([text_input])
        prediction = model.predict(X_input)[0]
        # Map prediction to label
        label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        predicted_label = label_map.get(prediction, "Unknown")
        st.subheader("Prediction Result")
        st.write(f"The predicted sentiment is: **{predicted_label}**")
    else:
        st.warning("Please enter some text.")

# Display instructions
st.write("""
### Instructions
1. Enter the text you want to analyze in the box above.
2. Click the 'Predict Sentiment' button to see the predicted sentiment.

""")
