# Normalize Text Program 4NL3 Homework 1

## Setup

To set up and run this program, you'll need to install the `NLTK` library. You can do this using the following command:

```bash
pip install nltk
```

## Running the program
To run the program, use the following command:

```bash
python normalize_text.py largetokens.txt (<options>)
```
Where `largetokens.txt` is the input text file, and `<options>` are any of the following available flags:

- `--lower`: Converts all text to lowercase.
- `--stem`: Apply stemming to the tokens in the text.
- `--lem`: Apply lemmatization to the tokens in the text.
- `--rem_stopwords`: Remove stopwords from the token list.
- `--min_len`: Specify the minimum length of tokens to keep. For example, `--min_len 3` will keep words that are 3 letters or more.
- `--log_y`: Use a logarithmic scale for the y-axis in the plot.

### Input File Format
The input file should be a plain text file with the .txt extension.

### Output
The program generates figures displaying the top 30 tokens based on various normalization options.

## Figures Directory
This directory contains the generated figures showing the top 30 tokens with different normalization options applied.

## Example
```bash
python normalize_text.py largetokens.txt --lower --stem --rem_stopwords
```
This command will process the text in `largetokens.txt`, convert all text to lowercase, apply stemming, and remove stopwords from the token list. Plot it on a graph and display the top 30 tokens.
