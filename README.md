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
├── app.py                      # The main Streamlit web application
├── requirements.txt            # List of Python dependencies
│
├── Dataset & Notebook/         # Training data and experiments
│   ├── Fake.csv                # Raw dataset (Fake News)
│   ├── True.csv                # Raw dataset (Real News)
│   └── model.ipynb             # Jupyter Notebook for EDA & Training
│
└── models/                     # Saved model artifacts (.pkl files)
    ├── model_2_fake_detection.pkl
    ├── model_subject_classification.pkl
    ├── vectorizer.pkl
    ├── label_encoder.pkl
    └── porter_stemmer.pkl
```

Note: The .pkl files in the models/ directory are generated after running model.ipynb.

🚀 Features

    Dual-Model Architecture:

        Authenticity Check: Distinguishes between trusted reporting (e.g., Reuters) and fabricated news.

        Topic Classification: Automatically categorizes articles into subjects like Politics, World News, or US News.

    High Accuracy: Achieves ~98.7% accuracy on fake news detection and ~92.5% on subject classification.

    Text Processing: Implements advanced cleaning including Regex filtering, Lowercasing, and Porter Stemming.

    Interactive UI: A user-friendly web interface built with Streamlit.

🛠️ Tech Stack

    Core: Python 3.x

    Data Processing: Pandas, NumPy, NLTK (Natural Language Toolkit)

    Machine Learning: Scikit-Learn (Logistic Regression, TF-IDF Vectorizer)

    Visualization: Matplotlib, Seaborn

    Deployment: Streamlit

💿 Installation & Usage
1. Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.
Bash

# Clone or download this repository
cd NEWS_DATASET

2. Install Dependencies

Install the required libraries using the provided requirements.txt file.
Bash

pip install -r requirements.txt

3. Train the Models (Optional)

If the models/ directory is empty, you need to train the models first.

    Open Dataset & Notebook/model.ipynb in Jupyter Notebook or VS Code.

    Run all cells. This will process Fake.csv and True.csv and generate the .pkl files using joblib.

    Move the generated .pkl files into the models/ folder if they are saved elsewhere.

4. Run the Web Application

Launch the Streamlit app to test news articles interactively.
Bash

streamlit run app.py

The app will open in your browser at http://localhost:8501.
🧠 Technical Details
Data Preprocessing

Raw text data is noisy. The model.ipynb notebook applies the following cleaning pipeline:

    Regex Cleaning: Removes non-alphabetical characters.

    Lowercasing: Standardizes text case.

    Stopword Removal: Removes common words (e.g., "the", "is") using NLTK.

    Stemming: Reduces words to their root form (e.g., "voting" → "vote") using PorterStemmer.

    Vectorization: Converts text to numerical format using TfidfVectorizer.

Model Performance

The models were trained on the ISOT Fake News Dataset (~45k articles).
Model	Algorithm	Training Accuracy	Testing Accuracy
Fake News Detector	Logistic Regression	99.2%	98.7%
Subject Classifier	Logistic Regression (Multinomial)	95.8%	92.5%
