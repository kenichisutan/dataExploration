import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import spacy

from preprocess import preprocess_text, lemmatize_text, get_noun_phrases, load_additional_stop_words


# Step 1: Feature Extraction for Clustering using TF-IDF
def extract_features(texts):
    stop_words_list = list(stop_words)  # Convert stop_words set to list
    vectorizer = TfidfVectorizer(stop_words=stop_words_list, max_df=0.5, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


# Step 2: Clustering using K-means
def apply_clustering(tfidf_matrix, num_clusters=5):
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    return clusters, km


# Modified load_and_process_data to return raw texts and article_ids as well
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
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        entities_list.append(entities)

    return processed_texts, raw_text_list, article_id_list, entities_list


# Step 3: LDA for topic modeling
def lda(texts, name, stop_words):
    print(f"Starting LDA for {name}")

    vectorizer = CountVectorizer(stop_words=list(stop_words))
    X = vectorizer.fit_transform(texts)

    lda_model = LatentDirichletAllocation(n_components=10, random_state=0)
    lda_model.fit(X)

    # Get the topic distribution for each document
    topic_distribution = lda_model.transform(X)

    # Print topic distribution for each document
    for i, topic_dist in enumerate(topic_distribution):
        print(f"Document {i}:")
        for topic_idx, topic_val in enumerate(topic_dist):
            print(f"  Topic {topic_idx}: {topic_val}")

    return lda_model, vectorizer, topic_distribution


# Function to get top words for each topic
def get_top_words(model, feature_names, n_top_words):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return top_words


# LDA function that processes each cluster separately
def lda_on_clusters(raw_texts, clusters, article_ids, name):
    cluster_texts = [[] for _ in range(max(clusters) + 1)]
    cluster_article_ids = [[] for _ in range(max(clusters) + 1)]

    for text, cluster, article_id in zip(raw_texts, clusters, article_ids):
        cluster_texts[cluster].append(text)
        cluster_article_ids[cluster].append(article_id)

    # Save cluster assignments to CSV
    cluster_df = pd.DataFrame({'article_id': article_ids, 'text': raw_texts, 'cluster': clusters})
    cluster_df.to_csv(f'{name}_cluster_assignments.csv', index=False)

    dominant_topics = []

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

    # Save dominant topics to a text file
    with open(f'{name}_dominant_topics.txt', 'w') as f:
        for article_id, cluster_id, topic_id in dominant_topics:
            f.write(f"Article ID: {article_id}, Cluster: {cluster_id}, Dominant Topic: {topic_id}\n")

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
    # LDA for ABC dataset
    all_text, raw_text, article_ids, entities_list = load_and_process_data(
        ['/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/ABC/ABC.csv'])
    tfidf_matrix, vectorizer = extract_features(raw_text)
    clusters, km = apply_clustering(tfidf_matrix, num_clusters=10)
    lda_on_clusters(raw_text, clusters, article_ids, 'ABC')
    silhouette_avg = silhouette_score(tfidf_matrix, clusters)
    print(f"Silhouette Score: {silhouette_avg}")

    # Save entities to a text file
    with open('recognized_entities.txt', 'w') as f:
        for article_id, entities in zip(article_ids, entities_list):
            f.write(f"Article ID: {article_id}\n")
            f.write(f"Entities: {entities}\n\n")

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load the stop words
additional_stop_words = load_additional_stop_words('../additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)
main()
