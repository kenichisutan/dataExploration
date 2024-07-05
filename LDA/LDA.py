import os
import string
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download the necessary NLTK data files
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# Step 1: Introduce pre-processing and lemmatization
def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

def lemmatize_text(text):
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def load_and_process_data(paths):
    all_text_list = []
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['content'] = df['content'].astype(str)
            all_text_list.extend(df['content'].tolist())
        else:
            print(f"File not found: {path}")

    # Preprocess and lemmatize the text
    all_text_list = [lemmatize_text(preprocess_text(text)) for text in all_text_list]
    return all_text_list

# Step 2: Introduce LDA
def lda(all_text, name):
    print(f"Starting LDA for {name}")

    vectorizer = CountVectorizer(stop_words=list(stop_words))
    X = vectorizer.fit_transform(all_text)

    lda = LatentDirichletAllocation(n_components=10, random_state=0)
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

    plt.figure(figsize=(20, 15))
    entity_freq_df.nlargest(30, 'frequency').plot(kind='bar', x='entity', y='frequency', legend=False)
    plt.xlabel('Entities', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Top 30 Entities in {name.upper()}', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig(f'./{name}_top_30_entities.png')
    plt.close()

    print(f"Finished LDA for {name}")

# Step 4: Run the code
stop_words = set(stopwords.words('english'))
additional_stop_words = ['would', 'could', 'get', 'like', '-', 'one', 'also', 'think', 'much', 'know', 'said', 'going', 'abc', 'want', 'back', 'dont', 'even', 'see', 'well', 'really', 'many', 'news', 'mr', 'new', 'fox', 'cnn', 'bbc', 'said', 'say', 'year', 'years', 'people']
stop_words.update(additional_stop_words)

all_text = load_and_process_data(['/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/articles_full.csv'])
# Run LDA
lda(all_text, 'articles')

# Step 5: Run the code on only ABC news
all_text = load_and_process_data(['/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/ABC/ABC.csv'])
lda(all_text, 'ABC')
