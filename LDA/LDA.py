import os
import string
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
import nltk
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download the necessary NLTK data files
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')

# Define functions for preprocessing and noun phrase extraction
def preprocess_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(words)

def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def get_noun_phrases(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # POS tagging
    pos_tags = pos_tag(words)

    # Define a chunking grammar
    grammar = r"""
      NP: {<DT>?<JJ>*<NN.*>+}    # Noun phrase with optional determiner and adjectives
          {<NNP>+}              # Proper noun sequences (e.g., "Donald Trump")
          {<NNP><NNP>}          # Two consecutive proper nouns (e.g., "Supreme Court")
          {<NN><NN>}            # Two consecutive nouns (e.g., "data science")
          {<JJ><NN>}            # Adjective followed by a noun (e.g., "big data")
          {<NNP><NNP><NNP>}     # Three consecutive proper nouns (e.g., "New York City")
    """

    # Create a chunk parser
    chunk_parser = nltk.RegexpParser(grammar)

    # Chunk the POS-tagged words
    chunked = chunk_parser.parse(pos_tags)

    # Convert chunked tree to IOB tags
    iob_tagged = tree2conlltags(chunked)

    noun_phrases = []
    current_np = []

    # Iterate over the IOB-tagged words
    for word, pos, chunk in iob_tagged:
        if chunk == 'B-NP' or chunk == 'I-NP':  # If the word is part of a noun phrase
            current_np.append(word)
        elif current_np:  # If we encounter a non-noun phrase and we have accumulated words
            noun_phrases.append('_'.join(current_np).lower())  # Join with underscores and lowercase
            current_np = []

    # If there are any remaining words in current_np, add them as a noun phrase
    if current_np:
        noun_phrases.append('_'.join(current_np).lower())

    return noun_phrases

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
        preprocessed_text = preprocess_text(text)
        lemmatized_text = lemmatize_text(preprocessed_text)
        noun_phrases = get_noun_phrases(lemmatized_text)
        processed_texts.append(' '.join(noun_phrases))

    return processed_texts

# Define the LDA function
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

# Run LDA on full articles dataset
all_text = load_and_process_data(['/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/articles_full.csv'])
lda(all_text, 'articles')

# Run LDA on ABC news dataset
all_text = load_and_process_data(['/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/articles/ABC/ABC.csv'])
lda(all_text, 'ABC')
