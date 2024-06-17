from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

df_comments = pd.read_csv('~/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/comments/ABC/ABC_all_comments.csv')

# Display dataset information
print(df_comments.head())
print(df_comments.info())
print(df_comments.describe())