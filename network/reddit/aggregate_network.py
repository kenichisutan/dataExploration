import pandas as pd

def aggregate_network(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Initialize the aggregation fields
    aggregation = {
        'article_id': lambda x: list(set(x)),
        'reddit_id': lambda x: list(set(x)),
        'parent_id': lambda x: list(set(x)),
        'entity_type': lambda x: list(set(x)),
        'sentiment': ['sum', lambda x: sum(val for val in x if val > 0), lambda x: sum(val for val in x if val < 0), lambda x: sum(1 for val in x if val > 0), lambda x: sum(1 for val in x if val < 0)],
        'timestamp': lambda x: list(set(x)),
    }

    # Group by 'entity', 'cluster', and 'topic' and aggregate the data
    aggregated_df = df.groupby(['entity', 'cluster', 'topic']).agg(aggregation).reset_index()

    # Rename the columns to match the desired output
    aggregated_df.columns = [
        'entity', 'cluster', 'topic', 'article_id', 'reddit_id', 'parent_id', 'entity_type',
        'sum_sentiment', 'sum_pos', 'sum_neg', 'weight_pos', 'weight_neg', 'timestamp'
    ]

    # Calculate the weight (total occurrences of the combination)
    aggregated_df['weight'] = aggregated_df['weight_pos'] + aggregated_df['weight_neg']

    # Reorder columns if needed
    aggregated_df = aggregated_df[['entity', 'cluster', 'topic', 'article_id', 'reddit_id', 'parent_id', 'entity_type',
                                   'sum_sentiment', 'sum_pos', 'sum_neg', 'weight_pos', 'weight_neg', 'weight', 'timestamp']]

    # Save the aggregated data to a new CSV file
    aggregated_df.to_csv(output_file, index=False)
    print(f"Aggregated network saved to {output_file}")

if __name__ == "__main__":
    input_file = 'reddit_entity_topic_network.csv'
    output_file = 'aggregated_reddit_entity_topic_network.csv'
    aggregate_network(input_file, output_file)
