# comparison between different outlets for the same platform
from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, word_tokenize
import string
import os
from wordcloud import WordCloud
import nltk

# Download the necessary NLTK data files
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


def load_additional_stop_words(file_path):
    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words


# Load the stop words
additional_stop_words = load_additional_stop_words('../additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)


# Preprocessing functions
def preprocess_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(words)


def lemmatize_text(text):
    words = word_tokenize(text)
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

    processed_texts = []
    for text in all_text_list:
        if text.strip():  # Check if the text is not empty
            preprocessed_text = preprocess_text(text)
            lemmatized_text = lemmatize_text(preprocessed_text)
            processed_texts.append(lemmatized_text)
        else:
            print(f"Skipping empty text in file")

    return ' '.join(processed_texts)


def most_common_words(all_text, name):
    print(f"Starting 20 most common words for {name}")
    all_text = all_text.translate(str.maketrans('', '', string.punctuation))

    words = all_text.split()
    word_freq = Counter(words)
    word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency'])

    plt.figure(figsize=(20, 15))
    word_freq_df.nlargest(20, 'frequency').plot(kind='bar', x='word', y='frequency', legend=False)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'20 Most Common Words in {name.upper()}', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig(f'./{name}_20_most_common_words.png')
    plt.close()
    print(f"Finished 20 most common words for {name}")


def word_cloud(all_text, name):
    print(f"Starting word cloud for {name}")
    all_text = all_text.translate(str.maketrans('', '', string.punctuation))

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate(all_text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'./{name}_word_cloud.png')
    plt.close()
    print(f"Finished word cloud for {name}")


base_path = os.path.expanduser('~/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data')
outlets = ['ABC', 'FOX', 'HIL', 'HP', 'MW', 'NW', 'NYT', 'WSJ']
platforms = ['articles', 'comments', 'reddit', 'twitter']

for platform in platforms:
    paths = []
    for outlet in outlets:
        if platform == 'articles':
            path = os.path.join(base_path, platform, outlet, f'{outlet}.csv')
        elif platform == 'reddit':
            path = os.path.join(base_path, platform, outlet, f'{outlet}_all_reddits.csv')
        elif platform == 'twitter':
            path = os.path.join(base_path, platform, outlet, f'{outlet}_all_tweets.csv')
        else:
            path = os.path.join(base_path, platform, outlet, f'{outlet}_all_{platform}.csv')
        paths.append(path)
        print(f"Checking path: {path}")
    all_text = load_and_process_data(paths)
    most_common_words(all_text, platform)
    word_cloud(all_text, platform)
