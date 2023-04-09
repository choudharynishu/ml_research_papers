'''This script is an implementation of character level Recurrent Neural Network on names data'''
from __future__ import unicode_literals, print_function, division
# Required Imports
import os
import sys
import glob
import string
import unicodedata
import argparse

import random

import torch
import torch.nn as nn
import torch.optim as optim


# torch.manual_seed(19949)
# ----------Model Architecture------------ #
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        """
        Weights automatically initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        """
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, last_hidden):
        combined = torch.cat((x, last_hidden), 1)
        #hidden = self.i2h(combined)
        #output = self.i2o(combined)
        hidden = torch.tanh(self.i2h(combined))
        output = torch.tanh(self.i2o(combined))
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# ----------File Extraction------------ #
def findFiles(path):
    return glob.glob(path)


# all_letters is a string containing all alphabets
# (both lower and uppercases) and whitespace, period,
# comma, semicolon, and single quotation mark
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Reference: https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(name):
    """
    :param name: a name (word) - a unicode string
    :return: converts the name to plain ascii, e.g. Vàsquez to Vasquez
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


print(unicodeToAscii('Vásquez'))


# Convert a unicode string (à, è, ñ)to ASCII
def readLines(filename):
    """
    :param filename: path to a file with names in a particular language
    :return: names (lines) converted back to ASCII
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


languages = []  # list of all langugages
language_label = {}  # dictionary for names in each language e.g., Spa
for file in findFiles(os.path.join(sys.path[0], 'names/*.txt')):
    language = os.path.splitext(os.path.basename(file))[0]
    languages.append(language)
    lines = readLines(file)
    language_label[language] = lines
n_languages = len(languages)

# ----------Converting extracted data into Tensors------------ #
# Code to convert alphabets (ascii) to number -one hot encoding vectors
def letterToIndex(letter):
    """
    :param letter: character (string)
    :return: character's position in the all_letters vocabulary
    """
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn an entire name into tensor of shape (name_length, 1, n_letters)
def lineToTensor(line):
    """
    :param line: a name (word) string type
    :return: a tensor of shape (len(name), 1, n_letters) with each 'column'
            representing one-hot encoding of a particular letter in the word
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, char in enumerate(line):
        tensor[i][0][letterToIndex(char)] = 1
    return tensor

def randomChoice(l):
    """
    :param l: List from which a random item is to be
                selected, here it represent list of all languages
    :return: random language string type
    """
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    """
    Here,
    languages : List of all languages
    line: a surname , string type
    language tensor: A tensor representing the index of randomly selected language
    line_tensor: A tensor of shape (1, len(line)) (len(line) = length of surname)
    :return:
    """
    language_i = randomChoice(languages)
    line = randomChoice(language_label[language_i])
    language_tensor = torch.tensor([languages.index(language_i)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return language_i, line, language_tensor, line_tensor

# ----------Training Model------------ #
# Initialize the model
n_hidden = 128  # convert later to user argument
rnn = RNN(n_letters, n_hidden, n_languages)
loss_func = nn.NLLLoss()
learning_rate = 0.01
max_iter = 10
current_loss = 0
loss_values = []


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    language_i = top_i[0].item()
    return languages[language_i], language_i


def training(language_tensor, line_tensor):
    hidden = rnn.initHidden()  # get initial hidden state as zero tensor with shape(1, hidden_size)

    # Define an optimizer instead
    rnn.zero_grad()  # zero-in the grad
    # line_tensor.size()[0] = surname length
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = loss_func(output, language_tensor)
    loss.backward()

    #  edit code below for optimizer.step()
    for parameter in rnn.parameters():
        # parameter.data._add(self, other, alpha) In-place version of add()
        # If both alpha and other are specified, each element of other is scaled by alpha before being used.
        parameter.data.add_(parameter.grad.data, alpha=learning_rate)
    return output, loss.item()


for iter in range(1, max_iter + 1):
    language_i, line, language_tensor, line_tensor = randomTrainingExample()
    output, loss = training(language_tensor, line_tensor)
    #current_loss += loss - plot later

    # Print iter number, loss, name and guess
    if iter % 1000 == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == language_i else '✗ (%s)' % language_i
        print(f"iteration: {iter}, loss:{loss}, line:{line}, guess:{guess}, correct:{correct} ")
        #print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
