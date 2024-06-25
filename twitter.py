from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

df_reddit = pd.read_csv('~/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/twitter/ABC/ABC_all_tweets.csv')

# Display dataset information
print(df_reddit.head())
print(df_reddit.info())
print(df_reddit.describe())

# Replace NaNs and non-string values with empty strings
df_reddit['content'] = df_reddit['content'].fillna('').astype(str)

# Define a function to preprocess text by removing stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the 'content' column
df_reddit['content'] = df_reddit['content'].apply(preprocess_text)

# Concatenate all text data
all_text = ' '.join(df_reddit['content'])

# Tokenize words
words = all_text.split()

# Count word frequencies
word_freq = Counter(words)

# Convert to DataFrame for visualization
word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency'])

# Plot 20 most common words
plt.figure(figsize=(10, 6))

word_freq_df.nlargest(20, 'frequency').plot(kind='bar', x='word', y='frequency', title='20 Most Common Words')
plt.savefig('twitter/20_most_common_words.png')
plt.close()

# Word Cloud
wordcloud = WordCloud().generate(all_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.savefig('twitter/wordcloud.png')
plt.close()

# Document Length Analysis
df_reddit['text_length'] = df_reddit['content'].apply(len)

# Plot distribution of text lengths
plt.figure(figsize=(10, 6))
df_reddit['text_length'].plot(kind='hist', bins=50, title='Text Length Distribution')
plt.close()
plt.savefig('twitter/text_length_distribution.png')
