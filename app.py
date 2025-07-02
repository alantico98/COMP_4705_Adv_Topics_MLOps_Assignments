import streamlit as st
import pandas as pd
import joblib



# --- Streamlit app layout ---
# 1. Title
st.title("Movie Review Sentiment Analyzer")

# 2. Description
st.write("""
         This app uses a Naive Bayes machine learning model to predict whether a movie reviews as **positive** or **negative**.
         """)

# 3. Load the Saved Model
# --- Caching the model ---
# Use st.cache_data to load the model only once
@st.cache_data
def load_model():
    model = joblib.load('./sentiment_model.pkl')
    model_labels = joblib.load('./sentiment_labels.pkl')
    return model, model_labels

# Load the model and class names
model, model_names = load_model()

# 4. Create the User Input Interface
movie_review = st.text_area("Enter a movie review to analyze:")

# 5. Predict and Show Results
if st.button("Analyze"):
    if movie_review:
        # Make prediction
        prediction = model.predict([movie_review])[0]
        probabilities = model.predict_proba([movie_review])[0]
        
        positive_score = round(probabilities[1] * 100, 2)
        negative_score = round(probabilities[0] * 100, 2)
        
        # Display the result
        if prediction == 'positive':
            st.success(f"The review is **Positive**")
            st.write(f"Confidence Probability: {positive_score}% positive, {negative_score}% negative")
        else:
            st.error(f"The review is **Negative**")
            st.write(f"Confidence Probability: {negative_score}% negative, {positive_score}% positive")
    else:
        st.error("Please enter a movie review to analyze.")