import os
import string
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from preprocess import preprocess_text, lemmatize_text, get_noun_phrases, load_additional_stop_words

# Download the necessary NLTK data files
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')

def load_and_process_data(paths):
    all_text_list = []
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

    return processed_texts

# Define the LDA function
def lda(all_text, name, stop_words):
    print(f"Starting LDA for {name}")

    vectorizer = CountVectorizer(stop_words=list(stop_words))
    X = vectorizer.fit_transform(all_text)

    lda = LatentDirichletAllocation(n_components=80, random_state=0)
    lda.fit(X)

    topic_words = []
    with open(f'{name}_topics.txt', 'w') as f:
        for i, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
            topic_words.append(top_words)
            f.write(f"Top 10 words for topic {i}:\n")
            f.write(", ".join(top_words) + "\n\n")
            print(f"Top 10 words for topic {i}")
            print(top_words)
            print()

    # Step 3: Find top 30 entities through LDA
    print(f"Top 30 entities for {name}")
    entity_list = []
    for i, topic in enumerate(lda.components_):
        entity_list.extend([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-30:]])
    entity_freq = Counter(entity_list)
    entity_freq_df = pd.DataFrame(entity_freq.items(), columns=['entity', 'frequency'])

    # Filter out stop words and meaningless entities from the result
    entity_freq_df = entity_freq_df[~entity_freq_df['entity'].isin(stop_words)]

    plt.figure(figsize=(20, 10))
    entity_freq_df.nlargest(30, 'frequency').plot(kind='bar', x='entity', y='frequency', legend=False)
    plt.xlabel('Entities', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Top 30 Entities in {name.upper()}', fontsize=22, pad=20)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)  # Adjust bottom to make space for labels
    plt.savefig(f'./{name}_top_30_entities.png')
    plt.close()

    print(f"Finished LDA for {name}")

def main():
    # Run LDA on full articles dataset
    all_text = load_and_process_data([
        '/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/articles_full.csv'])
    lda(all_text, 'articles', stop_words)

    # Run LDA on ABC news dataset
    all_text = load_and_process_data([
        '/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/ABC/ABC.csv'])
    lda(all_text, 'ABC', stop_words)


# Load the stop words
additional_stop_words = load_additional_stop_words('../additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)

if __name__ == "__main__":
    main()
