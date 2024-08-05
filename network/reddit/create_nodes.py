import pandas as pd

def load_data():
    # Load the CSV files
    dominant_topics_df = pd.read_csv('../ABC_dominant_topics.csv')
    reddit_entity_sentiment_df = pd.read_csv('reddit_entity_sentiment_analysis.csv')
    article_date_df = pd.read_csv('reddit_id_downloaded_date.csv')

    print("Loaded dataframes:")
    print("dominant_topics_df columns:", dominant_topics_df.columns)
    print("reddit_entity_sentiment_df columns:", reddit_entity_sentiment_df.columns)
    print("article_date_df columns:", article_date_df.columns)

    return dominant_topics_df, reddit_entity_sentiment_df, article_date_df

def merge_data(dominant_topics_df, reddit_entity_sentiment_df, article_date_df):
    # Merge the data on article_id
    merged_df = reddit_entity_sentiment_df.merge(dominant_topics_df, on='article_id', how='left')
    merged_df = merged_df.merge(article_date_df, on='reddit_id', how='left')
    return merged_df

def rename_and_reorder_columns(merged_df):
    # Rename columns
    merged_df = merged_df.rename(columns={'cluster_id': 'cluster', 'dominant_topic': 'topic'})

    # Reorder columns
    reordered_cols = ['article_id', 'reddit_id', 'parent_id', 'entity', 'entity_type', 'cluster', 'topic', 'sentiment', 'timestamp']
    merged_df = merged_df[reordered_cols]

    return merged_df

def save_network_to_csv(nodes_df, output_path):
    # Save the network to CSV
    nodes_df.to_csv(output_path, index=False)
    print(f"Network saved to {output_path}")

def main():
    # Load the data
    dominant_topics_df, reddit_entity_sentiment_df, article_date_df = load_data()

    # Merge the data
    merged_df = merge_data(dominant_topics_df, reddit_entity_sentiment_df, article_date_df)

    # Rename and reorder columns
    nodes_df = rename_and_reorder_columns(merged_df)

    # Save the network to CSV
    save_network_to_csv(nodes_df, 'reddit_entity_topic_network.csv')

if __name__ == "__main__":
    main()
