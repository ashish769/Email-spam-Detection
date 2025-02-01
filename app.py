import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("📧 Spam Detector")

# Create a text area for user input
user_input = st.text_area("Enter your email message here:")

# Button to check if the input is spam or not
if st.button("Check Spam"):
    input_vector = vectorizer.transform([user_input])  # Convert user input into the vectorized format
    prediction = model.predict(input_vector)[0]  # Predict the result
    if prediction == 1:
        st.error("🚨 This email is **SPAM**!")
    else:
        st.success("✅ This email is **NOT SPAM**!")
