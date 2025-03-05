# Corpus Normalization 4NL3 Homework 2

## Setup

To set up and run this program, you'll need to install the `numpy`, `pandas`, `nltk`, `gensim`, `scikit-learn`, `pyLDAvis` and `request`  libraries. You can do this using the following command:

```bash
pip install numpy pandas nltk gensim scikit-learn pyLDAvis requests
```

## Datasets
The program uses the following datasets from the `Harry-Potter` and `Percy-Jackson` directories, which have been divided into chapters using the `divide_into_chapters.py` script. If you want the original corpus, then they are in the `Harry Potter Series` and `Percy Jackson Series` directory. The datasets are in the form of plain text files with the .txt extension. The dataset was collected from [Harry Potter](https://kvongcmehsanalibrary.wordpress.com/wp-content/uploads/2021/07/harrypotter.pdf) and [Percy Jackson](https://freebiebooks.weebly.com/uploads/1/2/0/5/120506090/rick_riordan_-_percy_jackson_the_complete_collection.pdf). For the stop words I used a [Github post](https://gist.github.com/sebleier/554280).

## Running the program
To run the program, use the following command:

```bash
python -u "path file"
```

## Output
The program has few features:
- BoW: Bag of Words which will print the top 30 tokens of each category using count or TFIDF vectorize.
- LLR: Log Likelihood Ratio which will print the top 10 tokens of each category using count or TFIDF vectorize as BoW.
- LDA: Latent Dirichlet Allocation which will print the table with the columns as the topics and the rows as the top 25 tokens of each topic. Also, it outputs a html file with the visualization of the topics that can be opened in a browser.
- Average Topic Distribution: This will print the average topic distribution of each category.

## Results
- Harry Potter emphasizes school life, professors, and magical elements.
- Percy Jackson is more mythology-driven, highlighting gods, heroes, and quests.
