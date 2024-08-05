import os
import spacy
import pandas as pd
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, WordNetLemmatizer
import re

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

def load_and_process_comments_data(paths):
    all_text_list = []
    raw_text_list = []  # To keep raw texts for sentiment analysis
    article_id_list = []  # To keep article_ids
    comment_id_list = []  # To keep comment_ids
    parent_id_list = []  # To keep parent_ids
    entities_list = []  # To keep recognized entities
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['content'] = df['content'].astype(str)
            all_text_list.extend(df['content'].tolist())
            raw_text_list.extend(df['content'].tolist())
            article_id_list.extend(df['article_id'].tolist())
            comment_id_list.extend(df['comment_id'].tolist())
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

    return processed_texts, raw_text_list, article_id_list, comment_id_list, parent_id_list, entities_list

def load_additional_stop_words(file_path):
    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']

def sentiment_analysis_on_comments(raw_texts, article_ids, comment_ids, parent_ids, entities_list):
    comment_entity_sentiment = []

    for text, article_id, comment_id, parent_id, entities in zip(raw_texts, article_ids, comment_ids, parent_ids, entities_list):
        for entity_text, entity_label, entity_sent in entities:
            sentiment = analyze_sentiment(entity_sent.text)
            comment_entity_sentiment.append({
                'article_id': article_id,
                'comment_id': comment_id,
                'parent_id': parent_id,
                'entity': entity_text,
                'entity_type': entity_label,
                'sentiment': sentiment
            })

    # Save sentiment analysis results to a CSV file
    comment_entity_sentiment_df = pd.DataFrame(comment_entity_sentiment)
    comment_entity_sentiment_df.to_csv('comment_entity_sentiment_analysis.csv', index=False)

def main():
    # Paths to the CSV files for comments
    comment_paths = [
        '/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/comments/ABC/ABC_comm_final.csv'
    ]

    # Load and process comment data
    processed_comment_texts, raw_comment_texts, article_ids, comment_ids, parent_ids, entities_list = load_and_process_comments_data(comment_paths)

    # Perform sentiment analysis on comments
    sentiment_analysis_on_comments(raw_comment_texts, article_ids, comment_ids, parent_ids, entities_list)

# Load stop words
additional_stop_words = load_additional_stop_words('../../additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)
main()
