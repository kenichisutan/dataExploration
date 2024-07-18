import os
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from LDA.LDA import lda
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


# Modified load_and_process_data to return raw texts as well
def load_and_process_data(paths):
    all_text_list = []
    raw_text_list = []  # To keep raw texts for clustering
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['content'] = df['content'].astype(str)
            all_text_list.extend(df['content'].tolist())
        else:
            print(f"File not found: {path}")

    processed_texts = []
    for text in all_text_list:
        preprocessed_text = preprocess_text(text, stop_words)
        lemmatized_text = lemmatize_text(preprocessed_text)
        noun_phrases = get_noun_phrases(lemmatized_text)
        processed_texts.append(' '.join(noun_phrases))
        raw_text_list.append(text)  # Append raw text for clustering

    return processed_texts, raw_text_list


# LDA function that processes each cluster separately
def lda_on_clusters(raw_texts, clusters, name):
    cluster_texts = [[] for _ in range(max(clusters) + 1)]
    for text, cluster in zip(raw_texts, clusters):
        cluster_texts[cluster].append(text)

    # Save cluster assignments to CSV
    cluster_df = pd.DataFrame({'text': raw_texts, 'cluster': clusters})
    cluster_df.to_csv(f'{name}_cluster_assignments.csv', index=False)

    for idx, texts in enumerate(cluster_texts):
        if texts:
            print(f"Running LDA for cluster {idx}")
            processed_texts = [preprocess_text(text, stop_words) for text in texts]
            processed_texts = [lemmatize_text(text) for text in processed_texts]
            noun_phrases_texts = [' '.join(get_noun_phrases(text)) for text in processed_texts]
            lda(noun_phrases_texts, f"{name}_cluster_{idx}", stop_words)

    # Save all clusters to a single text file
    with open(f'{name}_clusters.txt', 'w') as f:
        for cluster_id in cluster_df['cluster'].unique():
            cluster_texts_count = len(cluster_df[cluster_df['cluster'] == cluster_id])
            f.write(f"\nCluster {cluster_id} (Total Articles: {cluster_texts_count}):\n")
            print(f"\nCluster {cluster_id} (Total Articles: {cluster_texts_count}):\n")
            sample_texts = cluster_df[cluster_df['cluster'] == cluster_id]['text'].sample(3, random_state=42).tolist()
            for text in sample_texts:
                f.write(f"{text}\n\n")
                print(f" - {text[:100]}...")


def main():
    # LDA for ABC dataset
    all_text, raw_text = load_and_process_data(
        ['/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/ABC/ABC.csv'])
    tfidf_matrix, vectorizer = extract_features(raw_text)
    clusters, km = apply_clustering(tfidf_matrix, num_clusters=10)
    lda_on_clusters(raw_text, clusters, 'ABC')
    silhouette_avg = silhouette_score(tfidf_matrix, clusters)
    print(f"Silhouette Score: {silhouette_avg}")

# Load the stop words
additional_stop_words = load_additional_stop_words('../additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)
main()
