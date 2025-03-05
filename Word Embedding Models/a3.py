import os
import argparse
import nltk
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.utils import run_queries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


nltk.download('stopwords')

# Stage 1: Load and Preprocess the Dataset

# Load the dataset
dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True, split="train")
sample_size = 100000
sample = dataset[:sample_size]
print(f"Dataset loaded successfully. Sample size: {sample_size}")

# Preprocessing function
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = text.replace('\n', ' ')
    tokens = text.split()
    return [word for word in tokens if word not in stop_words]

# Process entire dataset
corpus = [preprocess_text(sample["text"]) for sample in dataset]
# print(corpus[0])

# Train different variations of Word2Vec models
print("Training CBOW model...")
cbow_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, sg=0, workers=4)

print("Training Skip-gram model...")
skipgram_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, sg=1, workers=4)


def get_similar_words(model, word, topn=10):
    try:
        if isinstance(model, Word2Vec):
        # If it's a Word2Vec model, we access the 'wv' property
            similar_words = model.wv.most_similar(word, topn=topn)
        else:
            similar_words = model.most_similar(word, topn=topn)
        return similar_words
    except KeyError:
        return f"'{word}' not found in vocabulary."

def word_arithmetic(model, positive=[], negative=[], topn=10):
    try:
        if isinstance(model, Word2Vec):
            result = model.wv.most_similar(positive=positive, negative=negative, topn=10)
        else:
            result = model.most_similar(positive=positive, negative=negative, topn=10)
        return result
    except KeyError as e:
        return f"Word not found in the Vocabulary: {e}"
    
def print_formatted_results(results, title=None):
    if isinstance(results, str):  # Handle error messages
        print(f"\n{results}")
        return
        
    print("\n" + "="*50)
    if title:
        print(f"{title}")
    
    print("-"*50)
    for i, (word, similarity) in enumerate(results, 1):
        print(f"{i}. {word:<20} | Similarity: {similarity:.4f}")
    print("="*50 + "\n")

# ----------------------------------------------------------------------------------------------------------------
# Loading Pretrained Models
# ----------------------------------------------------------------------------------------------------------------

def validate_file_path(file_path):
    if file_path is None:
        return False
        
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return False
        
    if not os.path.isfile(file_path):
        print(f"Error: Path is not a file: {file_path}")
        return False
        
    if not os.access(file_path, os.R_OK):
        print(f"Error: File is not readable: {file_path}")
        return False
    
    return True
        

parser = argparse.ArgumentParser(description='Word Embedding Comparison Tool')
parser.add_argument('-gl', '--glove', help='Path to GloVe embeddings file', default=None)
parser.add_argument('-ft', '--fasttext', help='Path to fastText embeddings file', default=None)
args = parser.parse_args()


if args.glove:
    if validate_file_path(args.glove):
        print(f"GloVe embeddings file validated: {args.glove}")
    else:
        print(f"Cannot proceed with GloVe embeddings - invalid file path: {args.glove}")

if args.fasttext:
    if validate_file_path(args.fasttext):
        print(f"fastText embeddings file validated: {args.fasttext}")
    else:
        print(f"Cannot proceed with fastText embeddings - invalid file path: {args.fasttext}")



def load_vectors(file_path, is_glove=False, limit=None):
    try:
        if limit is None:
            if is_glove:
                return KeyedVectors.load_word2vec_format(file_path, no_header=True)
            else:
                return KeyedVectors.load_word2vec_format(file_path)
        else:
            if is_glove:
                return KeyedVectors.load_word2vec_format(file_path, no_header=True, limit=limit)
            else:
                return KeyedVectors.load_word2vec_format(file_path, limit=limit)
    except Exception as e:
        print(f"Error loading vectors from {file_path}: {e}")
        return None

glove_model = None
fasttext_model = None

if args.glove:
    glove_model = load_vectors(args.glove, is_glove=True, limit=50000)

if args.fasttext:
    fasttext_model = load_vectors(args.fasttext, limit=50000)

# Compare word vectors across different models
def compare_word_vectors(positive, negative, cbow_model, skipgram_model, glove_model, fasttext_model, topn=10):
    models = {
        "CBOW (self-trained)": cbow_model,
        "Skip-gram (self-trained)": skipgram_model,
        "GloVe (pre-trained)": glove_model,
        "fastText (pre-trained)": fasttext_model
    }
    
    results = {}
    
    # Determine if this is a similarity query or word arithmetic
    is_similarity_query = len(positive) == 1 and len(negative) == 0
    
    if is_similarity_query:
        print("\n" + "="*80)
        print(f"COMPARISON OF SIMILAR WORDS TO '{positive[0].upper()}' ACROSS DIFFERENT EMBEDDINGS")
        print("="*80)
    else:
        formula = " + ".join(positive)
        if negative:
            formula += " - " + " - ".join(negative)
        
        print("\n" + "="*80)
        print(f"COMPARISON OF WORD ARITHMETIC: {formula}")
        print("="*80)
    
    # Get results from each model
    for model_name, model in models.items():
        try:
            if isinstance(model, Word2Vec):
                arithmetic_result = model.wv.most_similar(positive=positive, negative=negative, topn=topn)
            else:
                arithmetic_result = model.most_similar(positive=positive, negative=negative, topn=topn)
                
            results[model_name] = arithmetic_result
            
            print(f"\n{model_name}:")
            print("-"*50)
            for i, (result_word, similarity) in enumerate(arithmetic_result, 1):
                print(f"{i}. {result_word:<20} | Similarity: {similarity:.4f}")
        except KeyError as e:
            word = str(e).strip("'")
            print(f"\n{model_name}: Word '{word}' not found in vocabulary")
    
    return results


queries = [
    (["technology"], []),  # Words similar to "technology"
    (["music", "happy"], []), # music + happy
    (["dog"], ["cat"]), # dog - cat
    (["Paris", "Germany"], ["France"]),  # Paris + Germany - France
    (["Microsoft", "iPhone"], ["Windows"]),  # Microsoft + iPhone - Windows 
    (["man", "computer"], ["woman"]), # Man + Computer - Woman

]

# Run all queries
for positive, negative in queries:
    compare_word_vectors(positive, negative, cbow_model, skipgram_model, glove_model, fasttext_model, topn=10)


# ----------------------------------------------------------------------------------------------------------------
# Stage 2: Socioeconomic Bias in Intelligence Analysis
# ----------------------------------------------------------------------------------------------------------------

target_words_1 = [
    "rich", "wealthy", "upper-class", "elite", "successful", 
    "privilege", "luxury", "inheritance", "mansion", "affluence"
]

target_words_2 = [
    "poor", "poverty", "working-class", "struggling", "paycheck", 
    "broke", "disadvantaged", "minimum-wage", "unemployed", "hardship"
]

attribute_words_1 = [
    "smart", "intelligent", "genius", "bright", "sharp", 
    "brilliant", "knowledgeable", "wise", "logical", "articulate"
]

attribute_words_2 = [
    "dumb", "ignorant", "slow", "uneducated", "clueless", 
    "foolish", "naive", "simple", "incompetent", "inarticulate"
]


# Create a new Query object for the WEAT test on socioeconomic bias
bias_query = Query(
    target_sets=[target_words_1, target_words_2],
    attribute_sets=[attribute_words_1, attribute_words_2],
    target_sets_names=["Wealth-Associated Terms", "Low-Income-Associated Terms"],
    attribute_sets_names=["Intelligence-Associated Terms", "Lack of Intelligence Terms"]
)

# Convert trained models into WEFE-compatible formats
wefe_models = {
    "Skip-Gram": WordEmbeddingModel(skipgram_model.wv, "Skip-Gram"),
    "CBOW": WordEmbeddingModel(cbow_model.wv, "CBOW"),
}

if glove_model:
    wefe_models["GloVe"] = WordEmbeddingModel(glove_model, "GloVe")
if fasttext_model:
    wefe_models["FastText"] = WordEmbeddingModel(fasttext_model, "FastText")

# Run WEAT test
weat = WEAT()

wefe_results = run_queries(
    WEAT, [bias_query], list(wefe_models.values()),
    metric_params={'preprocessors': [{}, {'lowercase': True}]},
    warn_not_found_words=True
).T.round(2)

print("\nBias Results:\n", wefe_results)


# ----------------------------------------------------------------------------------------------------------------
# Stage 3: Text Classification
# ----------------------------------------------------------------------------------------------------------------


def load_corpus(folder_path, label):
    corpus = []
    labels = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = preprocess_text(text)
            corpus.append(" ".join(tokens))
            labels.append(label)
    return pd.DataFrame({"text": corpus, "label": labels})

# Load datasets
hp_corpus_df = load_corpus("Harry-Potter", 0)
pj_corpus_df = load_corpus("Percy-Jackson", 1)

# Combine datasets
corpus_df = pd.concat([hp_corpus_df, pj_corpus_df], ignore_index=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(corpus_df["text"], corpus_df["label"], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train logistic regression with TF-IDF features
bow_model = LogisticRegression(max_iter=1000)
bow_model.fit(X_train_bow, y_train)
y_pred_bow = bow_model.predict(X_test_bow)

# Evaluate TF-IDF model
accuracy_bow = accuracy_score(y_test, y_pred_bow)
f1_bow = f1_score(y_test, y_pred_bow, average='weighted')
print(f"TF-IDF Model - Accuracy: {accuracy_bow:.4f}, F1-Score: {f1_bow:.4f}")

# Train Word2Vec model
corpus_tokens = [text.split() for text in corpus_df["text"]]
w2v_model = Word2Vec(sentences=corpus_tokens, vector_size=100, window=5, min_count=1, workers=4)

# Mean pooled embeddings
def get_mean_embedding(text, model):
    words = text.split()
    embeddings = [model.wv[word] for word in words if word in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

X_train_emb = np.array([get_mean_embedding(text, w2v_model) for text in X_train])

# Train logistic regression with mean pooled embeddings
emb_model = LogisticRegression(max_iter=1000)
emb_model.fit(X_train_emb, y_train)
X_test_emb = np.array([get_mean_embedding(text, w2v_model) for text in X_test])
y_pred_emb = emb_model.predict(X_test_emb)

# Evaluate mean pooled embeddings model
accuracy_emb = accuracy_score(y_test, y_pred_emb)
f1_emb = f1_score(y_test, y_pred_emb, average='weighted')
print(f"Mean Pooled Embeddings Model - Accuracy: {accuracy_emb:.4f}, F1-Score: {f1_emb:.4f}")

# Compare results and discuss efficiency
print("\nComparison:")
print(f"TF-IDF Model - Accuracy: {accuracy_bow:.4f}, F1-Score: {f1_bow:.4f}")
print(f"Mean Pooled Embeddings Model - Accuracy: {accuracy_emb:.4f}, F1-Score: {f1_emb:.4f}")