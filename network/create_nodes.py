import pandas as pd

def load_data():
    # Load the CSV files
    dominant_topics_df = pd.read_csv('ABC_dominant_topics.csv')
    article_date_df = pd.read_csv('article_id_downloaded_date.csv')
    entity_topic_sentiment_df = pd.read_csv('entity_topic_sentiment_analysis.csv')

    return dominant_topics_df, article_date_df, entity_topic_sentiment_df

def merge_data(dominant_topics_df, article_date_df, entity_topic_sentiment_df):
    # Merge the data on article_id
    merged_df = entity_topic_sentiment_df.merge(dominant_topics_df, on='article_id').merge(article_date_df, on='article_id')
    return merged_df

def save_network_to_csv(merged_df, output_path):
    # Save the network to CSV
    merged_df.to_csv(output_path, index=False)
    print(f"Network saved to {output_path}")

def main():
    # Load the data
    dominant_topics_df, article_date_df, entity_topic_sentiment_df = load_data()

    # Merge the data
    merged_df = merge_data(dominant_topics_df, article_date_df, entity_topic_sentiment_df)

    # Remove unnecessary columns
    merged_df = merged_df.drop(columns=['cluster_id', 'dominant_topic'])

    # Save the network to CSV
    save_network_to_csv(merged_df, 'entity_topic_network.csv')

if __name__ == "__main__":
    main()
