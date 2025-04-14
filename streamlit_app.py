import streamlit as st
import joblib
import pandas as pd
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import time

# Download required NLTK resources
# filepath: c:\codes\pyth\drug_ai\Drug_AI\streamlit_app.py
import os
nltk_data_path = os.path.expanduser('~\\nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

import nltk

# Ensure NLTK resources are downloaded
try:
    stop = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
    stop = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)
    lemmatizer = WordNetLemmatizer()

# Load resources
MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH = 'model/tfidfvectorizer.pkl'
#DATA_PATH = 'data/drugsComTest_raw.csv'
DATA_PATH='data/drugsComTrain.csv'

# try:
#     vectorizer = joblib.load(TOKENIZER_PATH)
#     model = joblib.load(MODEL_PATH)
#     df = pd.read_csv(DATA_PATH)
# except FileNotFoundError as e:
#     st.error(f"Error: One or more required files not found: {e}")
#     st.stop()  # Stop execution if files are missing


@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = joblib.load(TOKENIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model

# @st.cache_data
# def load_data():
#     return pd.read_csv(DATA_PATH)

vectorizer, model = load_model_and_vectorizer()
# df = load_data()

# stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# def cleanText(raw_review):
#     review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
#     letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
#     words = letters_only.lower().split()
#     meaningful_words = [w for w in words if not w in stop]
#     lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
#     return ' '.join(lemmitize_words)

# filepath: c:\codes\pyth\drug_ai\Drug_AI\streamlit_app.py
# Precompile regex pattern
letters_only_pattern = re.compile('[^a-zA-Z]')
stop = set(stopwords.words('english'))  # Convert to set for faster lookups

def cleanText(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = letters_only_pattern.sub(' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if w not in stop]
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmitize_words)


# def top_drugs_extractor(condition, df):
#     df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(
#         by=['rating', 'usefulCount'], ascending=[False, False]
#     )
#     drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
#     return drug_lst

# filepath: c:\codes\pyth\drug_ai\Drug_AI\streamlit_app.py
@st.cache_data
def preprocess_dataframe(data_path):
    df = pd.read_csv(data_path)
    return df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)]

df = preprocess_dataframe(DATA_PATH)

def top_drugs_extractor(condition, df):
    df_top = df[df['condition'] == condition].sort_values(
        by=['rating', 'usefulCount'], ascending=[False, False]
    )
    return df_top['drugName'].head(3).tolist()

st.title("Disease Condition and Drug Recommendation")

raw_text = st.text_area("Enter your text here:", "")


if st.button("Predict"):
    if raw_text:
        with st.spinner("Processing..."):
        # Prediction logic here
            # Measure time for cleaning
            start_time = time.time()
            clean_text = cleanText(raw_text)
            cleaning_time = time.time() - start_time

            # Measure time for prediction
            start_time = time.time()
            clean_lst = [clean_text]
            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]
            prediction_time = time.time() - start_time

            # Measure time for finding top drugs
            start_time = time.time()
            top_drugs = top_drugs_extractor(predicted_cond, df)
            top_drugs_time = time.time() - start_time

        # Display results
        st.write(f"**Predicted Condition:** {predicted_cond}")
        if top_drugs:
            st.write("**Top Recommended Drugs:**")
            for drug in top_drugs:
                st.write(f"- {drug}")
        else:
            st.write("No top drugs found for this condition based on the criteria.")

        # Display timing information
        # st.write(f"**Time Taken:**")
        # st.write(f"- Cleaning: {cleaning_time:.4f} seconds")
        # st.write(f"- Prediction: {prediction_time:.4f} seconds")
        # st.write(f"- Finding Top Drugs: {top_drugs_time:.4f} seconds")
    else:
        st.write("Please enter text for prediction.")