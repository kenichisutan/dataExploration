# comparison between different platforms for the same outlet
from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import string
import os

from wordcloud import WordCloud


def load_additional_stop_words(file_path):
    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words


# Load the stop words
additional_stop_words = load_additional_stop_words('../additional_stop_words.txt')
stop_words = set(stopwords.words('english'))
stop_words.update(additional_stop_words)

def load_and_process_data(paths):
    all_text_list = []
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['content'] = df['content'].astype(str)
            all_text_list.extend(df['content'].tolist())
        else:
            print(f"File not found: {path}")
    return ' '.join(all_text_list)

def most_common_words(all_text, name):
    print(f"Starting 20 most common words for {name}")
    all_text = all_text.translate(str.maketrans('', '', string.punctuation))

    def preprocess_text(text):
        words = text.split()
        words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(words)

    all_text = preprocess_text(all_text)

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

    def preprocess_text(text):
        words = text.split()
        words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(words)

    all_text = preprocess_text(all_text)

    wordcloud = WordCloud().generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(f'./{name}_wordcloud.png')
    plt.close()
    print(f"Finished word cloud for {name}")

# Paths to CSV files
base_path = os.path.expanduser('~/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data')
outlets = ['ABC', 'FOX', 'HIL', 'HP', 'MW', 'NW', 'NYT', 'WSJ']
platforms = ['articles', 'comments', 'reddit', 'twitter']

for outlet in outlets:
    paths = []
    for platform in platforms:
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
    if all_text:
        most_common_words(all_text, outlet.lower())
        word_cloud(all_text, outlet.lower())
    else:
        print(f"No data found for {outlet}")