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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# ----------Model Architecture------------ #
class NPLM(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_size, embedding_size2):
        super().__init__()
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(self.context_size*self.embedding_size, embedding_size2)
        self.linear2 = nn.Linear(embedding_size2, vocab_size)

    def forward(self, x):
        x = self.embedding(x).view((-1, self.context_size*self.embedding_size))
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        log_probs = F.log_softmax(x, dim =1)
        return log_probs


# ---------Implement DataLoader class--------  #
class Brown_news(Dataset):
    # -----------Build Dataset--------------------#
    """
    ==================   ==============   ==============
    Attributes           Bengio et al.    Current(news words)
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
    def __init__(self, rare_freq, context, category='news'):
        self.rare_freq = rare_freq
        self.context = context
        self.category = category
        #  Assigning 'Unknown' token to less frequent words
        self.unknown = "<UNK>"

        news_words = brown.words(categories=self.category)
        news_words = [w for w in news_words if w.isalpha()]
        news_words = [w.lower() for w in news_words]
        freq_dist = nltk.FreqDist(news_words)
        vocabulary = set()

        for w in news_words:
            if freq_dist[w] > self.rare_freq:
                vocabulary.add(w)

        vocabulary = list(vocabulary)
        vocabulary.append(self.unknown)
        self.vocab_size = len(vocabulary)
        self.vocab_idx_dict = {word: i for i, word in enumerate(vocabulary)}  # word to index dictionary
        print(f"Total number of words in the {category} category: {len(news_words)}")
        print(f"vocabulary-size for this text corpus: {self.vocab_size}")

        self.sample_size = len(news_words) // (context)
        #self.sample_size = len(news_words[:100]) // (context + 1)
        print(self.sample_size)

        X = torch.empty((self.sample_size, context), dtype=torch.long)
        y = torch.empty((self.sample_size, 1), dtype=torch.long)
        rolling_window = []
        start_index = 0
        i = 0

        while i < self.sample_size:
            X[i] = self.wordsToTensor(news_words[start_index:start_index + self.context])
            y[i] = self.wordToTensor(news_words[start_index + self.context + 1])
            start_index += 1
            i += 1

        self.x = X
        self.y = y
        #print(f"N_samples: {self.X.shape[0]} and {self.sample_size}")

    def wordToTensor(self, word):
        tensor = torch.zeros(1, dtype=torch.long)
        try:
            tensor[0] = self.vocab_idx_dict[word]
        except KeyError:
            tensor[0] = self.vocab_idx_dict[self.unknown]
        return tensor

    def wordsToTensor(self, words):
        """
        :param words: list of words of length = context
        :return: tensor with one-hot encoding for each word
        """
        tensor_list = []
        for i, word in enumerate(words):
            tensor_list.append(self.wordToTensor(word))
        return torch.cat(tensor_list)


    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.sample_size

    def get_vocab_size(self):
        return self.vocab_size

if __name__ == '__main__':
    # ---------parse command line args--------- #
    parser = argparse.ArgumentParser(description="NPLM (Bengio et al. 2003)")
    # ---------Dataset parameters--------- #
    parser.add_argument('--rare_freq', type=int, default=3, help="Frequency of the words for them to be classified as rare")
    parser.add_argument('--cat', type=str, default='news', help="Category of words to experiment on")

    # ---------Model architecture parameters--------- #
    parser.add_argument('--context', type=int, default=4, help="Context size")
    parser.add_argument('--n-embd', type=int, default=30, help="Size of the embeddings/Lookup table")
    parser.add_argument('--n-embd2', type=int, default=64, help=" Size of second layer of NPLM")

    # ---------Optimization Parameters--------- #
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=2e-3, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay") #Later

    args = parser.parse_args()

    rare_freq = args.rare_freq
    context = args.context  # Number of words used to predict the next word
    category = args.cat

    embedding_size = args.n_embd  # convert later to user argument
    embedding_size2 = args.n_embd2

    learning_rate = args.learning_rate
    batch_size = args.batch_size

    # ---------Initialize the DataLoader----------- #
    news_data = Brown_news(rare_freq, context, category)
    vocab_size = news_data.get_vocab_size()
    dataloader = DataLoader(dataset=news_data, batch_size=batch_size, shuffle=True)

    # ---------Initialize the Model------------- #
    model = NPLM(vocab_size, context, embedding_size, embedding_size2)
    loss_func = nn.NLLLoss()  # Using Negative Log-likelihood Loss

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---------Training the Algorithm----------- #
    dataiter = iter(dataloader)
    data = next(dataiter)
    features, labels = data
    print(features.shape)
    print(labels.shape)
    max_iter = 10
    current_loss = 0
    loss_values = []
    #  Separate into training and testing






