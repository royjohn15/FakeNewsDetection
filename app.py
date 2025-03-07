import streamlit as st
import sqlite3
import os
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from passlib.hash import sha256_crypt
import pandas as pd
from collections import Counter

# Download NLTK data (run once when app starts)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load pre-trained models and TF-IDF vectorizer
model_dir = 'models'
tfidf = joblib.load(os.path.join(model_dir, 'tfidf.pkl'))
rf_model = joblib.load(os.path.join(model_dir, 'rf.pkl'))
rf_fll_model = joblib.load(os.path.join(model_dir, 'rf_fll.pkl'))

# Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function: clean_text
def clean_text(text, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'www\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])

    if use_stemming:
        text = ' '.join([stemmer.stem(word) for word in text.split()])

    if use_lemmatization:
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

# Feature extraction function: eval_empirical_proportion
def eval_empirical_proportion(text):
    words = [word for word in text.split() if word.strip()]
    first_letters = [word[0].upper() for word in words if word[0].upper() in list(string.ascii_uppercase)]
    letter_counts = Counter(first_letters)
    total_letters = sum(letter_counts.values()) or 1  # Avoid division by zero
    letter_proportions = {letter: 0 for letter in string.ascii_uppercase}
    for letter, count in letter_counts.items():
        letter_proportions[letter] = count / total_letters
    sorted_proportions = dict(sorted(letter_proportions.items(), key=lambda item: item[1], reverse=True))
    return pd.Series(list(sorted_proportions.values()))

# Database setup and functions
def init_db():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS saved_articles
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  title TEXT,
                  article TEXT,
                  metric REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    conn.commit()
    conn.close()

def add_user(username, password):
    hashed = sha256_crypt.hash(password)
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()

def verify_login(username, password):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        user_id, hashed = result
        if sha256_crypt.verify(password, hashed):
            return user_id
    return None

def add_saved_article(user_id, title, article, metric):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("INSERT INTO saved_articles (user_id, title, article, metric) VALUES (?, ?, ?, ?)",
              (user_id, title, article, metric))
    conn.commit()
    conn.close()

def get_saved_articles(user_id):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("SELECT title, article, metric, timestamp FROM saved_articles WHERE user_id = ?", (user_id,))
    articles = c.fetchall()
    conn.close()
    return articles

# Streamlit application
def main():
    # Initialize database
    init_db()

    # Manage user session state
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None

    # Sidebar menu
    menu_options = ["Login", "Sign Up"] if not st.session_state['user_id'] else ["Predict", "View Saved"]
    choice = st.sidebar.selectbox("Menu", menu_options)

    ### Login Page
    if choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            user_id = verify_login(username, password)
            if user_id:
                st.session_state['user_id'] = user_id
                st.success(f"Welcome, {username}!")
                st.rerun()  # Refresh to update menu
            else:
                st.error("Incorrect username or password")

    ### Sign Up Page
    elif choice == "Sign Up":
        st.subheader("Sign Up")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type='password')
        if st.button("Sign Up"):
            if add_user(new_username, new_password):
                st.success("Account created! Please log in.")
            else:
                st.error("Username already exists")

    ### Prediction Page
    elif choice == "Predict":
        st.subheader("Fake News Detection")
        title = st.text_input("News Title")
        article = st.text_area("News Article", height=200)
        if st.button("Predict"):
            if not title or not article:
                st.warning("Please enter both a title and an article.")
            else:
                with st.spinner("Analyzing..."):
                    # Combine and preprocess
                    combined_text = title + " " + article
                    cleaned_text = clean_text(combined_text)

                    # Prediction with rf.pkl (TF-IDF features)
                    tfidf_features = tfidf.transform([cleaned_text])
                    prob_real_rf = rf_model.predict_proba(tfidf_features)[0][0]  # Probability of class 0 (real)

                    # Prediction with rf_fll.pkl (empirical proportion features)
                    empirical_features = eval_empirical_proportion(cleaned_text).values.reshape(1, -1)
                    prob_real_fll = rf_fll_model.predict_proba(empirical_features)[0][0]  # Probability of class 0 (real)

                    # Final metric: average probability of being real
                    final_metric = (prob_real_fll + prob_real_rf)/2

                    # Store in session state for saving
                    st.session_state['title'] = title
                    st.session_state['article'] = article
                    st.session_state['metric'] = final_metric

                    st.write(f"**Fakeness Metric:** {final_metric:.2f}")
                    st.write("(0 = fully fake, 1 = fully real)")

        # Save option after prediction
        if 'metric' in st.session_state and st.button("Save Result"):
            add_saved_article(st.session_state['user_id'], st.session_state['title'],
                             st.session_state['article'], st.session_state['metric'])
            st.success("Result saved successfully!")

    ### View Saved Articles Page
    elif choice == "View Saved":
        st.subheader("Your Saved Articles")
        articles = get_saved_articles(st.session_state['user_id'])
        if articles:
            for idx, (title, article, metric, timestamp) in enumerate(articles, 1):
                st.write(f"**{idx}. Title:** {title}")
                st.write(f"**Article:** {article[:200]}..." if len(article) > 200 else article)
                st.write(f"**Metric:** {metric:.2f} (0 = fake, 1 = real)")
                st.write(f"**Saved on:** {timestamp}")
                st.write("---")
        else:
            st.info("No saved articles yet.")

if __name__ == '__main__':
    main()