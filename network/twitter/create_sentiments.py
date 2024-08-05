import os
import re
import spacy
import pandas as pd
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, WordNetLemmatizer

# Load pre-trained SpaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Define functions for preprocessing and noun phrase extraction
def preprocess_text(text, stop_words):
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(words)

def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def load_and_process_tweets_data(paths):
    all_text_list = []
    raw_text_list = []  # To keep raw texts for sentiment analysis
    article_id_list = []  # To keep article_ids
    tweet_id_list = []  # To keep tweet_ids
    user_id_list = []  # To keep user_ids
    parent_id_list = []  # To keep parent_ids
    entities_list = []  # To keep recognized entities
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['content'] = df['content'].astype(str)
            all_text_list.extend(df['content'].tolist())
            raw_text_list.extend(df['content'].tolist())
            article_id_list.extend(df['article_id'].tolist())
            tweet_id_list.extend(df['tweet_id'].tolist())
            user_id_list.extend(df['user'].tolist())
            parent_id_list.extend(df['parent_id'].tolist())
        else:
            print(f"File not found: {path}")

    processed_texts = []
    for text in all_text_list:
        preprocessed_text = preprocess_text(text, stop_words)
        lemmatized_text = lemmatize_text(preprocessed_text)
        processed_texts.append(lemmatized_text)

        # Recognize entities in the preprocessed text
        doc = nlp(preprocessed_text)
        entities = [(ent.text, ent.label_, ent.sent) for ent in doc.ents if ent.label_ in {'PERSON', 'NORP', 'ORG', 'GPE'}]
        entities_list.append(entities)

    return processed_texts, raw_text_list, article_id_list, tweet_id_list, user_id_list, parent_id_list, entities_list

def load_additional_stop_words(file_path):
    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']

def sentiment_analysis_on_tweets(raw_texts, article_ids, tweet_ids, user_ids, parent_ids, entities_list):
    tweet_entity_sentiment = []

    for text, article_id, tweet_id, user_id, parent_id, entities in zip(raw_texts, article_ids, tweet_ids, user_ids, parent_ids, entities_list):
        for entity_text, entity_label, entity_sent in entities:
            sentiment = analyze_sentiment(entity_sent.text)
            tweet_entity_sentiment.append({
                'article_id': article_id,
                'tweet_id': tweet_id,
                'user_id': user_id,
                'parent_id': parent_id,
                'entity': entity_text,
                'entity_type': entity_label,
                'sentiment': sentiment
            })

    # Save sentiment analysis results to a CSV file
    tweet_entity_sentiment_df = pd.DataFrame(tweet_entity_sentiment)
    tweet_entity_sentiment_df.to_csv('twitter_entity_sentiment_analysis.csv', index=False)

def main():
    # Paths to the CSV files for tweets
    tweet_paths = [
        '/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/twitter/ABC/ABC_tweet_final.csv',
    ]

    # Load and process tweet data
    processed_tweet_texts, raw_tweet_texts, article_ids, tweet_ids, user_ids, parent_ids, entities_list = load_and_process_tweets_data(tweet_paths)

    # Perform sentiment analysis on tweets
    sentiment_analysis_on_tweets(raw_tweet_texts, article_ids, tweet_ids, user_ids, parent_ids, entities_list)

# Load stop words
additional_stop_words = load_additional_stop_words('/home/kenich/PycharmProjects/dataExploration/additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)
main()
