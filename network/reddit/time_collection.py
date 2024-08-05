import pandas as pd
import os


def extract_reddit_id_and_date(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    # Read the source CSV file
    df = pd.read_csv(input_path)

    # Extract the 'reddit_id' and 'timestamp' columns
    if 'reddit_id' in df.columns and 'timestamp' in df.columns:
        output_df = df[['reddit_id', 'timestamp']]
    else:
        print("Required columns ('reddit_id' and 'timestamp') not found in the input file.")
        return

    # Save the extracted data to a new CSV file
    output_df.to_csv(output_path, index=False)
    print(f"Output saved to {output_path}")


def main():
    # Define the input and output file paths
    input_path = '/home/kenich/MultiLayrtET2_Project/Data/2_proccessed_data_and_analysis/data/selected_data/reddit/ABC/ABC_reddit_final.csv'
    output_path = 'reddit_id_downloaded_date.csv'

    # Extract reddit_id and downloaded_date
    extract_reddit_id_and_date(input_path, output_path)


if __name__ == "__main__":
    main()
