# Define functions for preprocessing and noun phrase extraction
import os

import nltk
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer, pos_tag, tree2conlltags


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
    # Tokenize the text into words
    words = word_tokenize(text)

    # POS tagging
    pos_tags = pos_tag(words)

    # Define a chunking grammar
    grammar = r"""
      NP: {<DT>?<JJ>*<NN.*>+}       # Noun phrase with optional determiner and adjectives
          {<NNP>+}                  # Proper noun sequences (e.g., "Donald Trump")
          {<NNP><NNP>}              # Two consecutive proper nouns (e.g., "Supreme Court")
          {<NN><NN>}                # Two consecutive nouns (e.g., "data science")
          {<JJ><NN>}                # Adjective followed by a noun (e.g., "big data")
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
            # Filter out phrases longer than 2 words
            if len(current_np) <= 2:
                noun_phrases.append('_'.join(current_np).lower())  # Join with underscores and lowercase
            current_np = []

    # If there are any remaining words in current_np, add them as a noun phrase
    if current_np and len(current_np) <= 2:
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


def load_additional_stop_words(file_path):
    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words