# Word Embedding Comparison Tool

## Overview

This project compares different word embedding models (CBOW, Skip-gram, GloVe, and fastText) using a dataset derived from Wikipedia. The project evaluates the performance of these models using logistic regression on mean pooled embeddings and TF-IDF features. Additionally, it includes a WEAT test to analyze socioeconomic bias in intelligence.

## Libraries to Download

To set up and run this program, you'll need to install the following libraries:

```bash
pip install numpy pandas nltk gensim scikit-learn wefe
```

## How Does the Program Run
 - Load and Preprocess the Dataset: The program loads a sample of 100,000 articles from the Wikipedia dataset and preprocesses the text by removing stopwords and non-alphanumeric characters.
- Train Word2Vec Models: The program trains CBOW and Skip-gram models on the preprocessed corpus.
- Load Pre-trained Models: Optionally, the program can load pre-trained GloVe and fastText models if provided.
- Evaluate Models: The program evaluates the models using logistic regression on mean pooled embeddings and TF-IDF features.
- Compare Results: The program compares the accuracy and F1-score of the models and discusses their efficiency.
- WEAT Test: The program runs a WEAT test to analyze socioeconomic bias in intelligence using the trained models.

## Files/Dataset Used
- Wikipedia Dataset: A sample of 100,000 articles from the Wikipedia dataset.
- Harry-Potter: Directory containing text files of the Harry Potter series.
- Percy-Jackson: Directory containing text files of the Percy Jackson series.
- (Glove.42B.300d.txt)[https://nlp.stanford.edu/projects/glove/]
- (wiki-news-300d-1M.vec)[https://fasttext.cc/docs/en/english-vectors.html]

## How to Run the Program
```bash
python a3.py -gl /path/to/glove.42B.300d.txt -ft /path/to/wiki-news-300d-1M-subword.vec
```



