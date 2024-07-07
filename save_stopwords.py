additional_stop_words = [
    'would', 'could', 'get', 'like', '-', 'one', 'also', 'think', 'much', 'know', 'said', 'going', 'abc',
    'want', 'back', 'dont', 'even', 'see', 'well', 'really', 'many', 'news', 'mr', 'new', 'fox', 'cnn',
    'bbc', 'said', 'say', 'year', 'years', 'people', 'report', 'week', 'time', 'help', 'day', 'month',
    'world', 'country', 'americans', 'biden', 'trump', 'europe', 'twitter', 'thursday', 'tuesday', 'city',
    'foxnews', 'trump', 'nytimes', 'biden', 'ukraine', 'russia', 'people', 'amp', 'us', 'war', 'via', 'gop',
    'fox', 'right', 'news', 'hunter', 'nyt', 'bbc', 'cnn', 'today', 'tonight', 'tomorrow', 'monday',
    'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'yesterday', 'nytimes', 'nyc',
    'administration'
]

# Save to file
with open('additional_stop_words.txt', 'w') as f:
    for word in additional_stop_words:
        f.write(word + '\n')