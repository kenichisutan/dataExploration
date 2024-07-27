import os
import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, WordNetLemmatizer

# Load pre-trained SpaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()


def preprocess_text(text, stop_words):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(words)


def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def get_noun_phrases(text):
    # Use SpaCy to process the text
    doc = nlp(text)
    noun_phrases = []

    # Extract noun chunks and named entities
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 2:  # Only include phrases with 1 or 2 words
            noun_phrases.append(chunk.text.replace(' ', '_').lower())

    for ent in doc.ents:
        if len(ent.text.split()) <= 2:  # Only include entities with 1 or 2 words
            noun_phrases.append(ent.text.replace(' ', '_').lower())

    return noun_phrases


def load_and_process_data(paths):
    all_text_list = []
    raw_text_list = []  # To keep raw texts for clustering
    article_id_list = []  # To keep article_ids
    entities_list = []  # To keep recognized entities
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['content'] = df['content'].astype(str)
            all_text_list.extend(df['content'].tolist())
            raw_text_list.extend(df['content'].tolist())
            article_id_list.extend(df['article_id'].tolist())
        else:
            print(f"File not found: {path}")

    processed_texts = []
    for text in all_text_list:
        preprocessed_text = preprocess_text(text, stop_words)
        lemmatized_text = lemmatize_text(preprocessed_text)
        noun_phrases = get_noun_phrases(lemmatized_text)
        processed_texts.append(' '.join(noun_phrases))

        # Recognize entities in the raw text
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entities_list.append(entities)

    return processed_texts, raw_text_list, article_id_list, entities_list


def load_additional_stop_words(file_path):
    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words


def extract_features(texts):
    stop_words_list = list(stop_words)  # Convert stop_words set to list
    vectorizer = TfidfVectorizer(stop_words=stop_words_list, max_df=0.5, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def apply_clustering(tfidf_matrix, num_clusters=5):
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    return clusters, km


def lda(texts, name, stop_words):
    print(f"Starting LDA for {name}")

    vectorizer = CountVectorizer(stop_words=list(stop_words))
    X = vectorizer.fit_transform(texts)

    lda_model = LatentDirichletAllocation(n_components=10, random_state=0)
    lda_model.fit(X)

    topic_distribution = lda_model.transform(X)
    return lda_model, vectorizer, topic_distribution


def get_top_words(model, feature_names, n_top_words):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return top_words


def analyze_sentiment(text):
    sentiment = sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']


def analyze_entities_and_topics(raw_texts, clusters, article_ids, entities_list, topic_distribution):
    entity_topic_sentiment = []

    for i, (text, cluster, article_id, entities) in enumerate(zip(raw_texts, clusters, article_ids, entities_list)):
        topic_idx = np.argmax(topic_distribution[i])
        sentiment = analyze_sentiment(text)
        for entity_text, entity_label in entities:
            entity_topic_sentiment.append({
                'article_id': article_id,
                'entity': entity_text,
                'entity_type': entity_label,
                'topic': topic_idx,
                'sentiment': sentiment
            })

    return entity_topic_sentiment


def main():
    # Paths to the CSV files
    paths = [
        '/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/ABC/ABC.csv'
    ]

    # Load and process data
    processed_texts, raw_texts, article_ids, entities_list = load_and_process_data(paths)

    # Feature extraction and clustering
    tfidf_matrix, vectorizer = extract_features(raw_texts)
    clusters, km = apply_clustering(tfidf_matrix, num_clusters=10)

    # LDA topic modeling
    lda_model, vectorizer, topic_distribution = lda(processed_texts, 'ABC', stop_words)

    # Analyze entities, topics, and sentiment
    entity_topic_sentiment = analyze_entities_and_topics(raw_texts, clusters, article_ids, entities_list,
                                                         topic_distribution)

    # Save to CSV
    entity_topic_sentiment_df = pd.DataFrame(entity_topic_sentiment)
    entity_topic_sentiment_df.to_csv('entity_topic_sentiment_analysis.csv', index=False)


# Load stop words
additional_stop_words = load_additional_stop_words('../additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)
main()
