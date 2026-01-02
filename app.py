import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# SETUP
st.set_page_config(page_title="AI News Classifier", page_icon="📰")


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# LOAD MODELS & TOOLS

@st.cache_resource
def load_resources():
    try:
        fake_model = joblib.load('models/model_fake_detection.pkl')
        subject_model = joblib.load('models/model_subject_classification.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        encoder = joblib.load('models/label_encoder.pkl')
        stemmer = joblib.load('models/porter_stemmer.pkl')
        return fake_model, subject_model, vectorizer, encoder, stemmer
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Make sure all .pkl files are in the same folder.")
        return None, None, None, None, None

fake_model, subject_model, vectorizer, subject_encoder, port_stem = load_resources()

# PREPROCESSING FUNCTION 

def stemming(content):
    stop_words = set(stopwords.words('english'))
    
 
    content = re.sub(r'^\s*.*\(reuters\)\s*-\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\bcnn\b', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\breuters\b', '', content, flags=re.IGNORECASE)
    
   
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# UI

st.title("📰 AI News Classifier")
st.write("Enter a news headline or article content below to verify its authenticity and category.")

user_input = st.text_area("News Text:", height=150, placeholder="Paste article text here...")

if st.button("Analyze News"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif not fake_model:
        st.error("Models failed to load. Check your .pkl files.")
    else:
        with st.spinner("Analyzing text patterns..."):
           
            cleaned_text = stemming(user_input)
            
            vec_input = vectorizer.transform([cleaned_text])
            
            fake_prediction = fake_model.predict(vec_input)[0]
            fake_proba = fake_model.predict_proba(vec_input)[0]
            confidence = fake_proba[fake_prediction] * 100
            
            
            subject_pred_id = subject_model.predict(vec_input)[0]
            subject_name = subject_encoder.inverse_transform([subject_pred_id])[0]

           
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Authenticity")
                if fake_prediction == 1:
                    st.success(f"✅ REAL News\n\nConfidence: {confidence:.1f}%")
                else:
                    st.error(f"🚨 FAKE News\n\nConfidence: {confidence:.1f}%")
            
            with col2:
                st.subheader("Category")
                st.info(f"📂 {subject_name.title()}")

            with st.expander("See how the AI read this text"):
                st.write("**Cleaned & Stemmed Input:**")
                st.caption(cleaned_text)