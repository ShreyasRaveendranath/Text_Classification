import base64
import streamlit as st
import plotly.express as px
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
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Set the page configuration with the title and background image
st.set_page_config(
    page_title="Condition and Drug Name Prediction",
    page_icon=":pill:",
    layout="centered",
    initial_sidebar_state="expanded"
)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Load the trained model and vectorizer
model = load(open('model.pkl', 'rb'))
vectorizer = load(open('vectorizer.pkl', 'rb'))

# Add welcome message
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: black; font-size: 50px;">Welcome to our app!</h1>
        <p style="color: silver; font-size: 18px;">Enter your review and we'll predict the condition and recommend drugs for you.</p>
    </div>
    """,
    unsafe_allow_html=True
)

df = px.data.iris()

@st.cache
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.gifer.com/94aW.gif");
background-size: 110%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
maxlength_script = """
<script>
    const textarea = document.querySelector('textarea');
    const maxlength = %d;

    textarea.addEventListener('input', function(event) {
        const currentValue = event.target.value;
        if (currentValue.length > maxlength) {
            event.target.value = currentValue.slice(0, maxlength);
        }
    });
</script>
"""

# Maximum character limit
max_length = 100

# Add the JavaScript code to the Streamlit app
st.write(maxlength_script % max_length, unsafe_allow_html=True)

# Create the text area with the maximum character limit
text = st.text_area('Enter your review:', height=5)
# Create predict button to predict condition and recommended drugs

button_style = '''
    <style>
        .predict-button {
            background-color: black;
            color: gold;
            border-color: red;
        }
    </style>
'''
st.markdown(button_style, unsafe_allow_html=True)

if st.markdown('<button class="predict-button">Predict</button>', unsafe_allow_html=True):
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
            time.sleep(0.5)

            # Make prediction using the model
            pred1 = model.predict(tfidf_review)[0]

        # Display predicted condition
        st.subheader("Condition:")
        st.markdown(f'<p style="color: gold;">{pred1}</p>', unsafe_allow_html=True)

        # Load the data and get recommended drugs
        df2 = pd.read_table('drugsCom_raw (1).tsv')
        drug_ratings = df2[df2["condition"] == pred1].groupby("drugName")["rating"].mean()
        recommended_drugs = drug_ratings.nlargest(3).index.tolist()

        # Display recommended drugs
        st.subheader("Recommended Drugs:")
        for i, drug in enumerate(recommended_drugs):
            st.write(f'<span style="color: gold;">{i+1}. {drug}</span>', unsafe_allow_html=True)
