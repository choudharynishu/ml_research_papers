'''
This is an implementation of LeCun, Bottou, Bengio, and Haffner 1998, popularly known as LeNet5
Reference: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
'''
#Required imports
import os
import sys
import json
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
