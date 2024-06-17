from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load dataset
df_articles = pd.read_csv('~/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/process_data/articles/articles_full.csv')

# Display dataset information
print(df_articles.head())
print(df_articles.info())
print(df_articles.describe())

# Replace NaNs and non-string values with empty strings
df_articles['content'] = df_articles['content'].fillna('').astype(str)

# Define a function to preprocess text by removing stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the 'content' column
df_articles['content'] = df_articles['content'].apply(preprocess_text)

# Concatenate all text data
all_text = ' '.join(df_articles['content'])

# Tokenize words
words = all_text.split()

# Count word frequencies
word_freq = Counter(words)

# Convert to DataFrame for visualization
word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency'])

# Plot 20 most common words
plt.figure(figsize=(10, 6))
word_freq_df.nlargest(20, 'frequency').plot(kind='bar', x='word', y='frequency', title='20 Most Common Words')
plt.savefig('articles/20_most_common_words.png')
plt.close()

# Word Cloud
wordcloud = WordCloud().generate(all_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('articles/wordcloud.png')
plt.close()

# Document Length Analysis
df_articles['text_length'] = df_articles['content'].apply(len)

# Plot distribution of text lengths
plt.figure(figsize=(10, 6))
df_articles['text_length'].plot(kind='hist', bins=50, title='Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.savefig('articles/text_length_distribution.png')
plt.close()
