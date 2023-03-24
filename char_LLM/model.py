'''This script is an implementation of character level Recurrent Neural Network on names data'''
from __future__ import unicode_literals, print_function, division
# Required Imports
import os
import sys
import glob
import string
import unicodedata
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


def findFiles(path):
    return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Reference: https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(name):
    return ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


print(unicodeToAscii('Vásquez'))


# Convert a unicode string (à, è, ñ)to ASCII
def readLines(filename):
    '''
    :param filename: path to a file with names in a particular language
    :return: names (lines) converted back to ASCII
    '''
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


languages = []
language_label = {}
for file in findFiles(os.path.join(sys.path[0], 'names/*.txt')):
    language = os.path.splitext(os.path.basename(file))[0]
    languages.append(language)
    lines = readLines(file)
    language_label[language] = lines
