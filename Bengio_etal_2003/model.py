"""
This is an implementation of
    Yoshua Bengio, Réjean Ducharme, Pascal Vincent,
    and Christian Janvin. 2003. A neural probabilistic language model.
    J. Mach. Learn. Res. 3, null (3/1/2003), 1137–1155.
Reference: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

The paper discusses experimental results on
1. Brown Corpus
2. Associated Press text (1995-1996)

This implementation only reproduces result on the Brown Corpus
"""

# Required Imports
import os
import sys
import glob
import random
import argparse

"""
If running nltk first time, one would need to download the data using
>>>import nltk
>>>nltk.download('brown')
This downloads data in /root which on MacOS is /Users/yourname/
"""
import nltk
from nltk.corpus import brown

import torch
import torch.nn as nn
import torch.optim as optim

# ----------Model Architecture------------ #
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, embedding_size2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(vocab_size*embedding_size, embedding_size2)
        self.linear2 = nn.Linear(embedding_size2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.tanh(x)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)

        return x

# -----------Build Dataset--------------------#
"""
==================   ==============   ==============
Attributes           Bengio et al.    Current(news)
==================   ==============   ==============
Total                1,181,041        83,562
Training             800,000          5
Validation           200,000          6
Test                 181,041
Vocabulary-size      16,383           2950
Rare word Freq       3                3
Context size         4                4
==================   ==============   ==============
Pre-processing steps
1. Get rid of punctuations
2. Convert all words to lowercase
3. Build Frequency distribution of the words
4. Remove rare words (with frequency<=rare_freq)
5. Build Vocabulary and get its size
"""
rare_freq = 3
context = 4  # Number of words used to predict the next word
embedding_size = 30

news_words = brown.words(categories='news')
news_words = [w for w in news_words if w.isalpha()]
news_words = [w.lower() for w in news_words]
freq_dist = nltk.FreqDist(news_words)
vocabulary = set()

for w in news_words:
    if freq_dist[w] > rare_freq:
        vocabulary.add(w)
vocabulary = list(vocabulary)
vocab_size = len(vocabulary)

vocab_idx_dict = {word: i for i, word in enumerate(vocabulary)}  # probably inefficient
# print(news_words[:10])
print(f"Total number of words in the news category: {len(news_words)}")
print(f"vocabulary-size for this text corpus: {vocab_size}")


def wordToTensor(word):
    tensor = torch.zeros(1, vocab_size)
    tensor[0][vocab_idx_dict[word]] = 1
    return tensor


def wordsToTensor(words):
    """
    :param words: list of words of length = context
    :return: tensor with one-hot encoding for each word
    """
    tensor = torch.zeros(len(words), 1, vocab_size)
    for i, word in enumerate(words):
        tensor[i][0][vocab_idx_dict[word]] = 1
    return tensor


sample_size = len(news_words[:20]) // (context + 1)
print(sample_size)
X = torch.empty((sample_size, context, 1, vocab_size))
y = torch.empty((sample_size, 1, vocab_size))
rolling_window = []
# Only 10 for trial - for testing
# X.shape = ([len(data), context, 1, vocab_size])
# y.shape = ([len(data), 1, vocab_size])
start_index = 0
i = 0
while i < sample_size:
    X[i] = wordsToTensor(news_words[start_index:start_index + context])
    y[i] = wordToTensor(news_words[start_index + context + 1])
    start_index += 1
    i += 1
