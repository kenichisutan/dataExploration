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


def read_topic_file(file_path):
    topics = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if lines[i].startswith("Top 10 words for topic"):
                    if i + 1 < len(lines):
                        topic_words = lines[i + 1].strip()
                        topics.append(topic_words)
    return topics


def compile_unique_topics(input_file, output_file):
    topics = read_topic_file(input_file)
    unique_topics = list(set(topics))

    with open(output_file, 'w') as file:
        file.write("Unique Topics:\n")
        print("Unique Topics:")
        for idx, topic in enumerate(unique_topics):
            file.write(f"Topic {idx + 1}:\n")
            file.write(f"{topic}\n\n")
            print(f"Topic {idx + 1}")
            print(topic)
            print()

    print(f"Unique topics compiled into {output_file}")


# Define the input file path and the output file path
input_file_path = '/home/kenich/PycharmProjects/dataExploration/LDA/ABC_topics.txt'
output_file_path = 'ABC_unique_topics.txt'

# Run the compilation script
compile_unique_topics(input_file_path, output_file_path)
