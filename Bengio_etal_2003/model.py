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

# ---------------------------------Build Dataset--------------------#
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
context = 4

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

vocab_idx_dict = {word: i for i, word in enumerate(vocabulary)} #probably inefficient

print(f"Total number of words in the news category: {len(news_words)}")
print(f"vocabulary-size for this text corpus: {vocab_size}")



