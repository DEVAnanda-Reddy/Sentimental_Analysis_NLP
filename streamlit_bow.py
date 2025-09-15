import streamlit as st
import pickle

# Load model and vectorizer (pre-trained & saved from Jupyter)
with open("sentiment_model_bow.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer_bow.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Sentiment labels
sentiment_labels = {1: "Positive", 0: "Negative"}

def main():
    st.title("Amazon Review Sentiment Analysis using BOW + Random Forest")

    # User input
    user_input = st.text_area("Enter your review:")

    if st.button("Predict Sentiment"):
        if user_input.strip():
            # Preprocess and predict
            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]

            # Show result
            if prediction == 1:
                st.success("✅ Positive Sentiment")
            else:
                st.error("❌ Negative Sentiment")
        else:
            st.warning("Please enter a review.")

if __name__ == "__main__":
    main()
