from huggingface_hub import hf_hub_download
import os
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
nltk_data_path = os.path.expanduser('~\\nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

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

# Hugging Face repository details
HF_REPO_ID = "Jilani001/Drug_AI_Models"  # Replace with your Hugging Face repo ID
MODEL_FILENAME = "passmodel.pkl"
TOKENIZER_FILENAME = "tfidfvectorizer.pkl"

# Local paths for the model and tokenizer
LOCAL_MODEL_PATH = "model/passmodel.pkl"
LOCAL_TOKENIZER_PATH = "model/tfidfvectorizer.pkl"


# Download model and tokenizer from Hugging Face if not found locally
@st.cache_resource
def load_model_and_vectorizer():
    # Check if local files exist
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_TOKENIZER_PATH):
        model_path = LOCAL_MODEL_PATH
        tokenizer_path = LOCAL_TOKENIZER_PATH
    else:
        # Inform users about the download process
        with st.spinner("Downloading model and tokenizer..."):
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
            tokenizer_path = hf_hub_download(repo_id=HF_REPO_ID, filename=TOKENIZER_FILENAME)
            st.success("Download complete!")
    
    # Load the model and tokenizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(tokenizer_path)
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

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

@st.cache_data
def preprocess_dataframe(data_path):
    df = pd.read_csv(data_path)
    return df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)]

DATA_PATH = 'data/drugsComTrain.csv'
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

    else:
        st.write("Please enter text for prediction.")