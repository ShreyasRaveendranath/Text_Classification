import streamlit as st
from PIL import Image
import pandas as pd
from pickle import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import time

# Set the page configuration with the title and background image
st.set_page_config(
    page_title="Condition and Drug Name Prediction",
    page_icon=":pill:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the trained model and vectorizer
model = load(open('model.pkl', 'rb'))
vectorizer = load(open('vectorizer.pkl', 'rb'))

# Add welcome message
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: orange; font-size: 50px;">Welcome to our app!</h1>
        <p style="font-size: 18px;">Enter your review and we'll predict the condition and recommend drugs for you.</p>
    </div>
    """,
    unsafe_allow_html=True
)
# Load the image
image = Image.open("background.jpg")

# Reduce the image size
max_size = (800, 600)  # Set the maximum size for the image
image.thumbnail(max_size)

# Display the image
st.image(image, use_column_width=True)

# Create text input for user to enter review
text = st.text_area('Enter your review:', height=25)

# Create predict button to predict condition and recommended drugs
if st.button('Predict'):
    if not text:
        st.warning("Please enter a review.")
    else:
        # Clean the input review
        stop = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()

        def review_to_words(raw_review):
            review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
            letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
            words = letters_only.lower().split()
            meaningful_words = [w for w in words if not w in stop]
            lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
            return ' '.join(lemmatized_words)

        clean_review = review_to_words(text)

        # Vectorize the cleaned review
        tfidf_review = vectorizer.transform([clean_review])

        # Display loading spinner animation
        with st.spinner('Predicting...'):
            # Simulate prediction delay
            time.sleep(3)

            # Make prediction using the model
            pred1 = model.predict(tfidf_review)[0]

        # Display predicted condition
        st.subheader("Condition:")
        st.info(pred1)

        # Load the data and get recommended drugs
        df2 = pd.read_table('drugsCom_raw (1).tsv')
        drug_ratings = df2[df2["condition"] == pred1].groupby("drugName")["rating"].mean()
        recommended_drugs = drug_ratings.nlargest(3).index.tolist()

        # Display recommended drugs
        st.subheader("Recommended Drugs:")
        for i, drug in enumerate(recommended_drugs):
            st.write(f"{i+1}. {drug}")