import pandas as pd

def aggregate_network(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Initialize the aggregation fields
    aggregation = {
        'article_id': 'first',
        'entity': 'first',
        'entity_type': 'first',
        'cluster': 'first',
        'topic': 'first',
        'sentiment': ['sum', lambda x: sum(val for val in x if val > 0), lambda x: sum(val for val in x if val < 0)],
        'downloaded_date': 'first',
    }

    # Group by 'node_id' and aggregate the data
    aggregated_df = df.groupby('node_id').agg({
        'article_id': 'first',
        'entity': 'first',
        'entity_type': 'first',
        'cluster': 'first',
        'topic': 'first',
        'sentiment': ['sum', lambda x: sum(val for val in x if val > 0), lambda x: sum(val for val in x if val < 0), lambda x: sum(1 for val in x if val > 0), lambda x: sum(1 for val in x if val < 0)],
        'downloaded_date': 'first',
    }).reset_index()

    # Rename the columns to match the desired output
    aggregated_df.columns = [
        'node_id', 'article_id', 'entity', 'entity_type', 'cluster', 'topic',
        'sum_sentiment', 'sum_pos', 'sum_neg', 'weight_pos', 'weight_neg',
        'downloaded_date'
    ]

    # Calculate the weight (total occurrences of the node_id)
    aggregated_df['weight'] = aggregated_df['weight_pos'] + aggregated_df['weight_neg']

    # Save the aggregated data to a new CSV file
    aggregated_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = 'entity_topic_network.csv'
    output_file = 'aggregated_entity_topic_network.csv'
    aggregate_network(input_file, output_file)
