# 📰 AI Fake News Classifier & Subject Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

An end-to-end Machine Learning project that identifies whether a news article is **Fake** or **Real** and classifies its **Subject** (e.g., Politics, World News). The project includes a Jupyter Notebook for training and a Streamlit web application for real-time inference.

## 📂 Folder Structure
The project is organized as follows:

```text
NEWS_DATASET/
│
├── app.py                      
├── requirements.txt            
│
├── Dataset & Notebook/         
│   ├── Fake.csv                
│   ├── True.csv                
│   └── model.ipynb             
│
└── models/                     
    ├── model_2_fake_detection.pkl
    ├── model_subject_classification.pkl
    ├── vectorizer.pkl
    ├── label_encoder.pkl
    └── porter_stemmer.pkl
```

🚀 Features

    - Dual-Model Architecture:
    - Authenticity Check: Distinguishes between trusted reporting (e.g., Reuters) and fabricated news.
    - Topic Classification: Automatically categorizes articles into subjects like Politics, World News, or US News.
    - High Accuracy: Achieves ~98.7% accuracy on fake news detection and ~92.5% on subject classification.
    - Text Processing: Implements advanced cleaning including Regex filtering, Lowercasing, and Porter Stemming.
    - Interactive UI: A user-friendly web interface built with Streamlit.

🛠️ Tech Stack

   - Core: Python 3.x
   - Data Processing: Pandas, NumPy, NLTK (Natural Language Toolkit)
   - Machine Learning: Scikit-Learn (Logistic Regression, TF-IDF Vectorizer)
   - Visualization: Matplotlib, Seaborn
   - Deployment: Streamlit

💿 Installation & Usage
1. Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.
