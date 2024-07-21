import os

def read_topic_file(file_path):
    topics = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 3):  # Assuming each topic block is 3 lines (title, words, blank)
                if i + 1 < len(lines):
                    topic_words = lines[i + 1].strip()
                    topics.append(topic_words)
    return topics

def compile_unique_topics(input_dir, output_file):
    cluster_topics = {}

    for cluster_idx in range(10):  # Assuming 10 clusters
        cluster_file = os.path.join(input_dir, f'ABC_cluster_{cluster_idx}_topics.txt')
        topics = read_topic_file(cluster_file)
        cluster_topics[cluster_idx] = topics

    unique_topics = {}
    for cluster_idx, topics in cluster_topics.items():
        for topic in topics:
            if topic not in unique_topics:
                unique_topics[topic] = cluster_idx

    with open(output_file, 'w') as file:
        for cluster_idx in range(10):
            file.write(f"Cluster {cluster_idx}:\n")
            print(f"Cluster {cluster_idx}:")
            cluster_unique_topics = [topic for topic, origin_cluster in unique_topics.items() if origin_cluster == cluster_idx]
            for idx, topic in enumerate(cluster_unique_topics):
                file.write(f"Topic {idx + 1}:\n")
                file.write(f"{topic}\n\n")
                print(f"Topic {idx + 1}")
                print(topic)
                print()
            file.write("\n")
            print("\n")

    print(f"Unique topics compiled into {output_file}")

# Define the input directory containing the cluster topic files and the output file path
input_directory = '/home/kenich/PycharmProjects/dataExploration/LDA_cluster_w_wikipedia'
output_file_path = 'ABC_unique_topics.txt'

# Run the compilation script
compile_unique_topics(input_directory, output_file_path)
