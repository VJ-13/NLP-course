# NLP Course Projects

This repository contains various projects related to Natural Language Processing (NLP). Below is an overview of each folder and the libraries used for each project.

## Folders and Contents

### Word Embedding Models
This folder contains a project that compares different word embedding models (CBOW, Skip-gram, GloVe, and fastText) using a dataset derived from Wikipedia. The project evaluates the performance of these models using logistic regression on mean pooled embeddings and TF-IDF features. Additionally, it includes a WEAT test to analyze socioeconomic bias in intelligence.

#### Libraries Used
- numpy
- pandas
- nltk
- gensim
- scikit-learn
- wefe

### Normalizing Text
This folder contains a program for normalizing text as part of a homework assignment. The program can convert text to lowercase, apply stemming and lemmatization, remove stopwords, and filter tokens by length. It generates figures displaying the top 30 tokens based on various normalization options.

#### Libraries Used
- nltk

### LDA Analysis
This folder contains a project for corpus normalization and topic modeling using Latent Dirichlet Allocation (LDA). The project uses datasets from the Harry Potter and Percy Jackson series, divided into chapters. It includes features like Bag of Words (BoW), Log Likelihood Ratio (LLR), and LDA topic modeling with visualization.

#### Libraries Used
- numpy
- pandas
- nltk
- gensim
- scikit-learn
- pyLDAvis
- requests

## How to Run the Programs
Each folder contains a README file with detailed instructions on how to set up and run the respective programs. Please refer to the README files in each folder for more information.