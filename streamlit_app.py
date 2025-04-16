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

# # Download required NLTK resources
# nltk_data_path = os.path.expanduser('~\\nltk_data')
# if not os.path.exists(nltk_data_path):
#     os.makedirs(nltk_data_path)

# nltk.data.path.append(nltk_data_path)

# try:
#     stop = set(stopwords.words('english'))
# except LookupError:
#     nltk.download('stopwords', download_dir=nltk_data_path)
#     stop = set(stopwords.words('english'))

# try:
#     lemmatizer = WordNetLemmatizer()
# except LookupError:
#     nltk.download('wordnet', download_dir=nltk_data_path)
#     lemmatizer = WordNetLemmatizer()

# Set a custom NLTK data directory
nltk_data_path = os.path.expanduser('~\\nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Ensure required NLTK resources are downloaded
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
stop=set(stopwords.words('english'))
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
lemmatizer=WordNetLemmatizer()

# Hugging Face repository details
HF_REPO_ID = "Jilani001/Drug_AI_Models"  # Replace with your Hugging Face repo ID
MODEL_FILENAME = "passmodel_quantized_compressed.pkl"
TOKENIZER_FILENAME = "tfidfvectorizer_compressed.pkl"

# Local paths for the model and tokenizer
# LOCAL_MODEL_PATH = "model/passmodel.pkl"
# LOCAL_TOKENIZER_PATH = "model/tfidfvectorizer.pkl"

LOCAL_MODEL_PATH = "model/passmodel_quantized_compressed.pkl"
LOCAL_TOKENIZER_PATH = "model/tfidfvectorizer_compressed.pkl"


# Download model and tokenizer from Hugging Face if not found locally
@st.cache_resource
def load_model_and_vectorizer():
    # Check if local files exist
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_TOKENIZER_PATH):
        model_path = LOCAL_MODEL_PATH
        tokenizer_path = LOCAL_TOKENIZER_PATH
        st.success("Model and tokenizer loaded from local files.")
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

# Add a sidebar or a section for project details
st.sidebar.title("About the Project")

# Dataset Information
st.sidebar.subheader("Dataset")
st.sidebar.write("""
- **Name**: Drug Review Dataset (Drugs.com)
- **Source**: [UCI ML Repository](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)
- **Description**: This dataset contains patient reviews on specific drugs along with related conditions and a 10-star patient rating system.
""")

# Training Details
st.sidebar.subheader("Training Details")
st.sidebar.write("""
- **Top Conditions Trained**: 20
- **Total data points in the top 20 conditions**: 94446
- **Algorithm Used**: Passive Aggressive Classifier
- **Tokenizer**: Trigram TF-IDF Vectorizer
- **Model Optimization**:
  - Quantized and Compressed
  - Reduced Vocabulary Size
""")
trained_conditions = ["Birth Control","Depression","Pain","Anxiety","Acne","Bipolar Disorder","Insomnia","Weight Loss","Obesity","ADHD","Diabetes, Type 2","Emergency Contraception","High Blood Pressure","Vaginal Yeast Infection","Abnormal Uterine Bleeding","Bowel Preparation","ibromyalgia","Smoking Cessation","Migraine","Anxiety and Stress"]
with st.sidebar.expander("Top 20 Conditions Trained"):
    st.write(trained_conditions
             )
# Add an expander for more details
with st.sidebar.expander("Click for More Details"):
    st.write("""
    - **TF-IDF Settings**:
      - `max_features=20000`
      - `ngram_range=(1, 3)`
      - `max_df=0.8`
    - **Model Compression**:
      - Used Joblib with `compress=3`
    - **Quantization**:
      - Rounded model coefficients to 2 decimal places
    """)


raw_text = st.text_area("Enter your medical condition/review:", "")
if st.button("Predict"):
    if raw_text:
        with st.spinner("Processing..."):
            # Clean the input text
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]
            
            # Predict the condition
            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]

            # Check if the predicted condition is in the trained list
            if predicted_cond not in trained_conditions:
                st.write("The entered condition is not supported by the model.")
            else:
                # Find top drugs for the predicted condition
                top_drugs = top_drugs_extractor(predicted_cond, df)
                st.write(f"**Predicted Condition:** {predicted_cond}")
                if top_drugs:
                    st.write("**Top Recommended Drugs:**")
                    for drug in top_drugs:
                        st.write(f"- {drug}")
                else:
                    st.write("No drugs found for the predicted condition. Please consult a healthcare professional.")
    else:
        st.write("Please enter text for prediction...")

# Add a main section for project introduction
st.markdown("""
This project predicts the **condition** of a patient based on their **drug reviews**. It uses a **Passive Aggressive Classifier** trained on the **Drug Review Dataset** from [UCI ML Repository](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018).

### Key Features:
- **Dataset**: Contains reviews, conditions, and ratings for various drugs.
- **Total dataset data points**: 160398       
- **Top Conditions Trained**: 20
- **Total data points in the top 20 conditions**: 94446
                   
- **Data Preprocessing**: Text cleaning, lemmatization, and stopword removal (NLP).
- **Model**: Passive Aggressive Classifier with TF-IDF Vectorization also deployed in [HuggingFace repo](https://huggingface.co/Jilani001/Drug_AI_Models/tree/main).
- **Data Filtering**: Only reviews with a rating of 9 or higher and at least 100 useful votes are considered.
- **Drug Recommendations**: Based on the predicted condition, the top 3 recommended drugs are displayed.
- **Performance**: The model is optimized for faster inference and reduced memory usage.
- **Model Optimization**:
  - Quantized and compressed for faster inference.
  - Reduced vocabulary size for efficiency.
- **User-Friendly Interface**: Built using Streamlit for easy interaction.
- **Deployment**: The app is deployed on Streamlit Cloud for public access.
-- **Future Work**: Potential for further optimization and expansion to include more conditions and drugs.
- **Limitations**: The model is trained on a specific subset of conditions and may not generalize well to all drug reviews.
            
""")