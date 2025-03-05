# Importing necessary libraries
import argparse # For parsing command-line arguments
import re # For regular expressions
import nltk # For natural language processing tasks
from nltk.corpus import stopwords # For stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer # For stemming and lemmatization
from collections import Counter # For counting token occurrences
import matplotlib.pyplot as plt # For plotting data

# NLTK data
nltk.download('stopwords', quiet=True) # Download stopwords
nltk.download('wordnet', quiet=True) # Download WordNet to run lemmatization
nltk.download('omw-1.4', quiet=True) # Download Open Multilingual WordNet to run lemmatization

# Function to preprocess text it takes in the file path, lower, stem, lem, rem_stopwords, min_len as arguments
def preprocess_text(file_path, lower = False, stem = False, lem = False, rem_stopwords = False, min_len = 1):

    # Load stopwords
    stop_words = set(stopwords.words("english")) if rem_stopwords else set()

    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Read the file with utf-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Makes the text lowercase if lower is True
    if lower:
        text = text.lower()

    # Replace hyphens with spaces to separate words joined by hyphens
    text = re.sub(r'[-]+', ' ', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize text
    tokens = text.split()

    # Apply preprocessing steps
    processed_tokens = []

    for token in tokens:
        # Skip the token if it is a stopword
        if rem_stopwords and token in stop_words:
            continue
        # Skip the token if it is shorter than the minimum length
        if len(token) < min_len:
            continue
        # Apply stemming
        if stem:
            token = stemmer.stem(token)
        # Apply lemmatization
        if lem:
            token = lemmatizer.lemmatize(token)
        # Append the token to the list of processed tokens
        processed_tokens.append(token)

    # Count occurrences
    token_counts = Counter(processed_tokens)

    # Return the token counts
    return token_counts

# Function to visualize token counts it takes in sorted_tokens and log_y as arguments
def visualize_token_counts(sorted_tokens, log_y=False):
    
    # Setup the ranks
    ranks = range(1, len(sorted_tokens) + 1)

    # Extract tokens and frequencies
    tokens = [token for token, _ in sorted_tokens]
    freq = [count for _, count in sorted_tokens]

    # Plot the data
    plt.figure(figsize = (15, 8))
    plt.bar(ranks, freq, tick_label=tokens)

    # Apply log scale if specified
    if log_y:
        plt.yscale('log')

    # Add labels and title
    plt.title("Token Frequency Visualization")
    plt.xlabel('Tokens')
    plt.ylabel('Frequency of Tokens')

    # Rotate x-axis labels for better readability
    plt.xticks(ticks=ranks, labels=tokens, rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

# Main function
def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input text file.")
    parser.add_argument("--lower", action="store_true", help="Convert text to lowercase.")
    parser.add_argument("--stem", action="store_true", help="Apply stemming to tokens.")
    parser.add_argument("--lem", action="store_true", help="Apply lemmatization to tokens.")
    parser.add_argument("--rem_stopwords", action="store_true", help="Remove stopwords.")
    parser.add_argument("--min_len", type=int, default=1, help="Minimum length of tokens to keep.")
    parser.add_argument("--log_y", action="store_true", help="Use log scale for the y-axis in the plot.")

    # Parse arguments
    args = parser.parse_args()

    # Process the text
    token_counts = preprocess_text(
        args.file,
        lower = args.lower,
        stem = args.stem,
        lem = args.lem,
        rem_stopwords = args.rem_stopwords,
        min_len = args.min_len
    )

    # Sort tokens by frequency and print top 30 tokens
    sorted_tokens = sorted(token_counts.items(), key = lambda x: x[1], reverse = True)[:30]
    for token, count in sorted_tokens:
        print(f"{token}: {count}")

    # Print total number of tokens
    print(f"Total Tokens: {len(token_counts)}")

    # Visualize token frequencies
    visualize_token_counts(sorted_tokens, log_y = args.log_y)

# Entry point of the script
if __name__ == "__main__":
    main()
