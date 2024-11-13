import streamlit as st
import joblib

# Load the pre-trained model and vectorizer
clf = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI
st.title("Movie Review Sentiment Classifier")

st.write(
    """
    This app classifies a movie review as either Positive or Negative.
    Enter a review below to see the classification.
    """
)

# Input field for movie review
user_review = st.text_area("Enter a Movie Review:")

if st.button("Classify Review"):
    if user_review:
        # Transform the input review using the loaded vectorizer
        review_transformed = vectorizer.transform([user_review])

        # Predict the sentiment
        prediction = clf.predict(review_transformed)

        # Display the result
        if prediction == 1:
            st.success("This movie review is POSITIVE!")
        else:
            st.error("This movie review is NEGATIVE.")
    else:
        st.warning("Please enter a review to classify.")
