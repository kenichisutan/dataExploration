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

# Define functions for preprocessing and noun phrase extraction
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
        entities = [(ent.text, ent.label_, ent.sent) for ent in doc.ents if ent.label_ in {'PERSON', 'NORP', 'ORG', 'GPE'}]
        entities_list.append(entities)

    return processed_texts, raw_text_list, article_id_list, entities_list

def load_additional_stop_words(file_path):
    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words

# Feature Extraction for Clustering using TF-IDF
def extract_features(texts):
    stop_words_list = list(stop_words)  # Convert stop_words set to list
    vectorizer = TfidfVectorizer(stop_words=stop_words_list, max_df=0.5, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

# Clustering using K-means
def apply_clustering(tfidf_matrix, num_clusters=5):
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    return clusters, km

# LDA for topic modeling
def lda(texts, name, stop_words):
    print(f"Starting LDA for {name}")

    vectorizer = CountVectorizer(stop_words=list(stop_words))
    X = vectorizer.fit_transform(texts)

    lda_model = LatentDirichletAllocation(n_components=10, random_state=0)
    lda_model.fit(X)

    topic_distribution = lda_model.transform(X)
    return lda_model, vectorizer, topic_distribution

# Function to get top words for each topic
def get_top_words(model, feature_names, n_top_words):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return top_words

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']

# LDA function that processes each cluster separately
def lda_on_clusters(raw_texts, clusters, article_ids, entities_list, name):
    cluster_texts = [[] for _ in range(max(clusters) + 1)]
    cluster_article_ids = [[] for _ in range(max(clusters) + 1)]
    cluster_entities_list = [[] for _ in range(max(clusters) + 1)]

    for text, cluster, article_id, entities in zip(raw_texts, clusters, article_ids, entities_list):
        cluster_texts[cluster].append(text)
        cluster_article_ids[cluster].append(article_id)
        cluster_entities_list[cluster].append(entities)

    # Save cluster assignments to CSV
    cluster_df = pd.DataFrame({'article_id': article_ids, 'text': raw_texts, 'cluster': clusters})
    cluster_df.to_csv(f'{name}_cluster_assignments.csv', index=False)

    dominant_topics = []
    entity_topic_sentiment = []

    for idx, texts in enumerate(cluster_texts):
        if texts:
            print(f"Running LDA for cluster {idx}")
            processed_texts = [preprocess_text(text, stop_words) for text in texts]
            processed_texts = [lemmatize_text(text) for text in processed_texts]
            noun_phrases_texts = [' '.join(get_noun_phrases(text)) for text in processed_texts]
            lda_model, vectorizer, topic_distribution = lda(noun_phrases_texts, f"{name}_cluster_{idx}", stop_words)

            # Find the dominant topic for each document in the cluster
            most_significant_topics = np.argmax(topic_distribution, axis=1)
            for article_id, topic_idx in zip(cluster_article_ids[idx], most_significant_topics):
                dominant_topics.append((article_id, idx, topic_idx))

            # Get the top words for each topic and save to file
            feature_names = vectorizer.get_feature_names_out()
            top_words = get_top_words(lda_model, feature_names, 10)
            with open(f'{name}_cluster_{idx}_topics.txt', 'w') as f:
                for topic_idx, words in top_words.items():
                    f.write(f"Topic {topic_idx}: {', '.join(words)}\n")

            # Analyze sentiment for each entity in the cluster and combine with entity and topic information
            for i, (text, article_id, entities) in enumerate(zip(texts, cluster_article_ids[idx], cluster_entities_list[idx])):
                topic_idx = most_significant_topics[i]
                for entity_text, entity_label, entity_sent in entities:
                    sentiment = analyze_sentiment(entity_sent.text)
                    entity_topic_sentiment.append({
                        'article_id': article_id,
                        'entity': entity_text,
                        'entity_type': entity_label,
                        'cluster': idx,
                        'topic': topic_idx,
                        'sentiment': sentiment
                    })

    # Save dominant topics to a CSV file
    dominant_topics_df = pd.DataFrame(dominant_topics, columns=['article_id', 'cluster_id', 'dominant_topic'])
    dominant_topics_df.to_csv(f'{name}_dominant_topics.csv', index=False)

    # Save sentiment analysis results to a CSV file
    entity_topic_sentiment_df = pd.DataFrame(entity_topic_sentiment)
    entity_topic_sentiment_df.to_csv('entity_topic_sentiment_analysis.csv', index=False)

    # Save all clusters to a single text file
    with open(f'{name}_clusters.txt', 'w') as f:
        sorted_clusters = sorted(cluster_df['cluster'].unique())
        for cluster_id in sorted_clusters:
            cluster_texts_count = len(cluster_df[cluster_df['cluster'] == cluster_id])
            f.write(f"\nCluster {cluster_id} (Total Articles: {cluster_texts_count}):\n")
            print(f"\nCluster {cluster_id} (Total Articles: {cluster_texts_count}):\n")
            sample_texts = cluster_df[cluster_df['cluster'] == cluster_id]['text'].sample(3, random_state=42).tolist()
            for text in sample_texts:
                f.write(f"{text}\n\n")
                print(f" - {text[:100]}...")

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

    # Perform LDA on each cluster and analyze entities, topics, and sentiment
    lda_on_clusters(raw_texts, clusters, article_ids, entities_list, 'ABC')

# Load stop words
additional_stop_words = load_additional_stop_words('../additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)
main()
