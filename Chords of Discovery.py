import streamlit as st
import pandas as pd
import nltk
import re
import spacy
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load SpaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Download NLTK stopwords and Vader lexicon
nltk.download('stopwords')
nltk.download('vader_lexicon')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Function to normalize text (remove special characters, stop words, and digits)
def normalize_document(doc):
    # Lowercase and remove special characters and whitespaces
    doc = re.sub(r"[^a-zA-Z\s']", '', doc, flags=re.I | re.A)
    doc = doc.lower().strip()

    # Tokenize and remove stop words
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words and not token.isdigit()]
    doc = ' '.join(filtered_tokens)
    doc = re.sub(r"'\s*", "", doc)  # Remove standalone apostrophes
    return doc

# Function to lemmatize text using the provided dataset logic
def lemmatize_text(text):
    # Normalize the input text
    normalized_text = normalize_document(text)
    
    # Create a SpaCy document for lemmatization
    doc = nlp(normalized_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

# Sentiment analysis logic using NLTK's SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(lyrics):
    sentiment = analyzer.polarity_scores(lyrics)
    sentiment_score = sentiment['compound']
    return sentiment_score

# Function to analyze sentiment of input song
def analyze_sentiment(song_title):
    lemmatized_title = lemmatize_text(song_title)
    sentiment_score = get_sentiment_score(lemmatized_title)
    
    if sentiment_score >= 0.05:
        sentiment = 'positive'
    elif sentiment_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment

def find_similar_songs(input_song, dataset_path='FullyProcessedDataset.csv', num_suggestions=3):
    """
    Find similar songs based on topic distribution and Hellinger distance.
    
    Args:
        input_song (str): Title of the input song.
        dataset_path (str): Path to the processed dataset.
        num_suggestions (int): Number of similar songs to return.

    Returns:
        list: Titles of similar songs.
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Ensure input song exists in the dataset
    if input_song not in df['SName'].values:
        return ["Input song not found in dataset."]
    
    # Extract input song's topic distribution (Assuming lda_model and corpus are predefined)
    song_topic_distribution = [lda_model[doc] for doc in corpus]  # Assuming this exists as in the notebook
    input_song_index = df.index[df['SName'] == input_song].item()
    input_song_dist = song_topic_distribution[input_song_index]
    
    # Calculate Hellinger distances for all songs
    distances = []
    for i, song_dist in enumerate(song_topic_distribution):
        distance = hellinger(input_song_dist, song_dist)
        distances.append((df['SName'].iloc[i], distance))
    
    # Sort by distance and return top results
    distances.sort(key=lambda x: x[1])  # Smallest distances first
    similar_songs = [title for title, _ in distances[:num_suggestions]]
    
    return similar_songs

# Streamlit app
def main():
    st.title("Music Recommendation System")
    st.write("Discover new music by entering your favorite song title.")

    # Input: Song title
    song_title = st.text_input("Enter a song title", "")
    
    if song_title:
        # Step 1: Preprocess input
        preprocessed_title = lemmatize_text(song_title)
        st.write(f"Lemmatized text: {preprocessed_title}")
        
        # Step 2: Perform Sentiment Analysis
        sentiment = analyze_sentiment(song_title)
        st.write(f"Sentiment of '{song_title}': {sentiment}")
        	
        # Step 3: Suggest similar songs
        dataset = [
            {"title": "Song X", "sentiment": "positive"},
            {"title": "Song Y", "sentiment": "neutral"},
            {"title": "Song Z", "sentiment": "negative"},
        ]
        similar_songs = find_similar_songs(song_title, dataset)
        st.write("Songs similar to your input:")
        for song in similar_songs:
            st.write(f"- {song}")

if __name__ == "__main__":
    main()
