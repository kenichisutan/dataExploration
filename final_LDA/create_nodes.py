import pandas as pd
import os


def load_data():
    # Load the CSV files
    dominant_topics_df = pd.read_csv('ABC_dominant_topics.csv')
    article_date_df = pd.read_csv('article_id_downloaded_date.csv')
    entity_topic_sentiment_df = pd.read_csv('entity_topic_sentiment_analysis.csv')

    return dominant_topics_df, article_date_df, entity_topic_sentiment_df


def merge_data(dominant_topics_df, article_date_df, entity_topic_sentiment_df):
    # Merge the data on article_id
    merged_df = entity_topic_sentiment_df.merge(dominant_topics_df, on='article_id').merge(article_date_df,
                                                                                           on='article_id')
    return merged_df


def create_nodes(merged_df):
    # Create nodes based on entity, cluster, and topic
    merged_df['node'] = merged_df.apply(lambda row: f"{row['entity']}_{row['cluster_id']}_{row['dominant_topic']}",
                                        axis=1)
    return merged_df


def generate_node_ids(nodes_df):
    # Generate unique node IDs
    nodes_df['node_id'] = nodes_df['node'].factorize()[0]

    # Reorder columns to have node_id as the leftmost column
    cols = ['node_id'] + [col for col in nodes_df.columns if col != 'node_id']
    nodes_df = nodes_df[cols]

    return nodes_df


def save_network_to_csv(nodes_df, output_path):
    # Save the network to CSV
    nodes_df.to_csv(output_path, index=False)
    print(f"Network saved to {output_path}")


def main():
    # Load the data
    dominant_topics_df, article_date_df, entity_topic_sentiment_df = load_data()

    # Merge the data
    merged_df = merge_data(dominant_topics_df, article_date_df, entity_topic_sentiment_df)

    # Create nodes
    nodes_df = create_nodes(merged_df)

    # Generate node IDs
    nodes_df = generate_node_ids(nodes_df)

    # Save the network to CSV
    save_network_to_csv(nodes_df, 'entity_topic_network.csv')


if __name__ == "__main__":
    main()
