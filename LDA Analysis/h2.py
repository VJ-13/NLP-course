# Importing necessary libraries

import os # For interacting with the file system
import re 
import math 
import requests 
import gensim # For topic modeling
import pyLDAvis # For visualizing LDA topics
import pyLDAvis.gensim_models as gensimvis 
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer # For TF-IDF
import numpy as np 
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer # For stemming and lemmatization
from collections import Counter # For counting token occurrences


# Download stopwords list from GitHub
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stop_words = set(stopwords_list.decode().splitlines()) 



# Set of character names from Harry Potter and Percy Jackson series
character_names = {
    "harry", "harrys", "ron", "weasley", "hermione", "dumbledore", "snape", "voldemort", "hagrid", "draco", "sirius", "lupin", "dobby", "umbridge", "potter",  # HP characters
    "percy", "annabeth", "grover", "poseidon", "zeus", "hades", "chiron", "tyson", "thalia", "nico", "luke", "silenus"  # PJ characters
}


# Function to preprocess text it takes in the file path, lower, stem, lem, rem_stopwords, min_len as arguments
def preprocess_text(file_path, lower = False, stem = False, lem = False, rem_stopwords = False, min_len = 1):
    
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Read the file with utf-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if lower:
        text = text.lower()

    # Removing non-alphanumeric characters
    text = re.sub(r'[-]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize text
    tokens = text.split()

    # Apply preprocessing steps
    processed_tokens = []

    for token in tokens:
        # Skip the token if it is a stopword
        if rem_stopwords and token in stop_words:
            continue
        # Remove character names
        if token in character_names:  
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

    # Return the list of processed tokens
    return processed_tokens

# Function to load and process documents from a folder
def load_corpus(folder_path, lower=True, stem=False, lem=False, rem_stopwords=True, bow_type = None, min_len = 1):
    corpus = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Process the file and add the tokens to the corpus
        if os.path.isfile(file_path):
            tokens = preprocess_text(file_path, lower, stem, lem, rem_stopwords, min_len)
            if bow_type == "count":
                corpus.extend(tokens)
            else:
                corpus.append(tokens)

    # Return the corpus as a bag of words, tfidf matrix or list of tokens
    if bow_type == "count":
        return Counter(corpus)
    elif bow_type == "tfidf":
        texts = [" ".join(doc) for doc in corpus]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    return corpus


# --------------------------------------------------
# --------------------------------------------------
# BoW for categories Harry Potter and Percy Jackson
# --------------------------------------------------
# --------------------------------------------------

# Load the BoW matrices with type count for Harry Potter and Percy Jackson
hp_bow_count = load_corpus("Harry-Potter", lower=True, stem=False, lem=False, rem_stopwords=True, bow_type="count")
pj_bow_count = load_corpus("Percy-Jackson", lower=True, stem=False, lem=False, rem_stopwords=True, bow_type="count")

# Load the BoW matrices with type tfidf for Harry Potter and Percy Jackson
hp_bow_tfidf = load_corpus("Harry-Potter", lower=True, stem=False, lem=False, rem_stopwords=True, bow_type="tfidf")
pj_bow_tfidf = load_corpus("Percy-Jackson", lower=True, stem=False, lem=False, rem_stopwords=True, bow_type="tfidf")

# Convert the BoW matrices to dictionaries
hp_bow_tfidf_dict = hp_bow_tfidf.sum(axis=0).to_dict()
pj_bow_tfidf_dict = pj_bow_tfidf.sum(axis=0).to_dict()

# --------------------------------------------------
# BoW using Count
# --------------------------------------------------

# print("Harry Potter Bag of Words:")
# hp_sorted_tokens = sorted(hp_bow_count.items(), key = lambda x: x[1], reverse = True)[:30]
# for token, count in hp_sorted_tokens:
#     print(f"{token}: {count}")

# print("\nPercy Jackson Bag of Words:")
# pj_sorted_tokens = sorted(pj_bow_count.items(), key = lambda x: x[1], reverse = True)[:30]
# for token, count in pj_sorted_tokens:
#     print(f"{token}: {count}")

# --------------------------------------------------
# BoW using Tfidf
# --------------------------------------------------

# print("Harry Potter Bag of Words:")
# hp_sorted_tokens = sorted(hp_bow_tfidf_dict.items(), key = lambda x: x[1], reverse = True)[:30]
# for token, count in hp_sorted_tokens:
#     print(f"{token}: {count}")

# print("\nPercy Jackson Bag of Words:")
# pj_sorted_tokens = sorted(pj_bow_tfidf_dict.items(), key = lambda x: x[1], reverse = True)[:30]
# for token, count in pj_sorted_tokens:
#     print(f"{token}: {count}")


# --------------------------------------------------
# --------------------------------------------------
# Computing LLR for Harry Potter and Percy Jackson categories
# --------------------------------------------------
# --------------------------------------------------


def compute_llr(bow_c1, bow_c2):
    # Find the total number of words in each category
    total_c1 = sum(bow_c1.values())
    total_c2 = sum(bow_c2.values())
    # Get the vocabulary for both categories
    vocab = set(bow_c1.keys()).union(set(bow_c2.keys()))
    llr = {}
    # Compute the LLR score for each word
    for word in vocab:
        p_wc1 = (bow_c1.get(word, 0) + 1) / (total_c1 + len(vocab))
        p_wc2 = (bow_c2.get(word, 0) + 1) / (total_c2 + len(vocab))
        llr[word] = round(math.log(p_wc1) - math.log(p_wc2), 2)

    # Return the top 10 words based on LLR score
    sorted_words = sorted(llr.items(), key=lambda x: x[1], reverse=True)[:10]
    return sorted_words

# --------------------------------------------------
# LLR using Count
# --------------------------------------------------

# print("Top 10 words for Harry Potter:")
# for word, score in compute_llr(hp_bow_count, pj_bow_count):
#     print(f"{word}: {score}")

# print("\nTop 10 words for Percy Jackson:")
# for word, score in compute_llr(pj_bow_count, hp_bow_count):
#     print(f"{word}: {score}")

# --------------------------------------------------
# LLR using Tfidf
# --------------------------------------------------


# print("Top 10 words for Harry Potter:")
# for word, score in compute_llr(hp_bow_tfidf_dict, pj_bow_tfidf_dict):
#     print(f"{word}: {score}")

# print("\nTop 10 words for Percy Jackson:")
# for word, score in compute_llr(pj_bow_tfidf_dict, hp_bow_tfidf_dict):
#     print(f"{word}: {score}")


# --------------------------------------------------
# --------------------------------------------------
# LDA Analysis for Harry Potter and Percy Jackson categories
# --------------------------------------------------
# --------------------------------------------------

# Load the corpus for both categories
hp_corpus = load_corpus("Harry-Potter", lower=True, stem=False, lem=False, rem_stopwords=True)
pj_corpus = load_corpus("Percy-Jackson", lower=True, stem=False, lem=False, rem_stopwords=True)
all_corpus = hp_corpus + pj_corpus

# Create a dictionary and BoW representation of the corpus
id2word = corpora.Dictionary(all_corpus)
texts = [id2word.doc2bow(text) for text in all_corpus]

# Train LDA model
num_topics = 10
# Compute the LDA by using the gensim library with arguments random_state for reproducibility, update_every for batch learning, chunksize for the number of documents to be used in each training chunk, passes for the number of passes through the corpus
lda_model = gensim.models.LdaModel(texts, num_topics=num_topics, id2word=id2word, random_state = 100, update_every=0, chunksize=20, passes=15)

# Visualize topics with mmds for better separation and R=25 for better readability
vis = gensimvis.prepare(lda_model, texts, id2word, sort_topics=False, mds="mmds", R=25)

# Save visualization to an HTML file
pyLDAvis.save_html(vis, "lda_visualization.html") 

# Define manual topic labels based on word distributions
manual_topic_labels = {
    0: "Dark Magic & Villains",
    1: "Heroes & Hunters",
    2: "School Life",
    3: "Gods & Divine Beings",
    4: "Water & Nature",
    5: "Professors & Authority",
    6: "Family & Demigod Origins",
    7: "Cyclops & Sea Creatures",
    8: "Ministry of Magic & Bureaucracy",
    9: "Competitions & Challenges"
}


# Extract top words for each topic with manual labels
def get_topics(lda_model, num_words=25):
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    topic_data = []
    # Iterate over each topic and extract the top words
    for topic_id, words in topics:
        topic_label = manual_topic_labels.get(topic_id, f"Topic {topic_id}")
        word_list = [f"{word[0]} ({word[1]:.4f})" for word in words]
        topic_data.append([topic_label] + word_list)

    # Return the topic data as a DataFrame to display as a table using pandas
    return pd.DataFrame(topic_data)

# Display topics as a table in the command line
topic_table = get_topics(lda_model)
print(topic_table)


# Compute average topic distribution for each category
def average_topic_distribution(lda_model, corpus, id2word=id2word, manual_topic_labels=manual_topic_labels):
    topic_distributions = []
    # Iterate over each document in the corpus and get the topic distribution
    for doc in corpus:
        bow = id2word.doc2bow(doc)
        topics = lda_model.get_document_topics(bow, minimum_probability=0)
        topic_distributions.append([prob for _, prob in topics])
    avg_distribution = np.mean(topic_distributions, axis=0)
    # Get the top 5 topics based on probability
    sorted_topics = sorted(enumerate(avg_distribution), key=lambda x: x[1], reverse=True)[:5]
    return [(manual_topic_labels.get(topic_id, f"Topic {topic_id}"), prob) for topic_id, prob in sorted_topics]

# Compute average topic distribution for Harry Potter and Percy Jackson categories
hp_top_topics = average_topic_distribution(lda_model, hp_corpus)
pj_top_topics = average_topic_distribution(lda_model, pj_corpus)

# Print the top topics for each category
print("Harry Potter Top Topics:", hp_top_topics)
print("Percy Jackson Top Topics:", pj_top_topics)