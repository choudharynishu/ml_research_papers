'''Many-to-Many Character level implementation of RNN from scratch'''
# Required Imports
import os
import sys
import shutil
from urllib.request import urlopen
import numpy as np

shakespear_data_url = r"https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
local_filename = f"shakespeare_input.txt"

file_path = os.path.join(sys.path[0], f"{local_filename}")
# Check if file already exists locally
if not os.path.exists(file_path):
    assert isinstance(shakespear_data_url, str), "Input url currently must be a string"
    with urlopen(shakespear_data_url) as response, open(file_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

data = open(file_path, 'r').read()
chars = list(set(data))
vocab_size = len(chars)
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
sequence_length = 40  # No of x-vectors to estimate the final hidden state
hidden_size = 100  # Number of neurons in the hidden layer of RNN (only one used here)
learning_rate = 0.01  # Standard learning-rate

# Initialize the parameters of the model
U = np.random.rand((hidden_size, vocab_size))
V = np.random.rand((hidden_size, vocab_size))
# General user inputs
total_epochs = 20
epoch, start_idx = 0, 0  # n = current number of iteration

# Define the loss function and update parameters
def loss_func(x, y, hidden_s):
    #Forward-pass
    #backward-pass
    loss_value = 0
    return loss_value, hidden_s

while epoch <= total_epochs:
    # After every epoch reset the RNN memory
    start_idx = 0
    hidden_state = np.zeros((hidden_size, 1))
    while start_idx + sequence_length + 1 < len(data):
        x_seq = [char_to_index[char] for char in data[start_idx:start_idx + sequence_length]]
        y_seq = [char_to_index[char] for char in data[start_idx + 1:start_idx + sequence_length + 1]]
        if epoch % 5 == 0:
            x = 1
            # Print the current value of loss
        # Forward pass
        '''
        Remember the shapes for sanity check
        U = (hidden_size, vocab_size)
        W = (hidden_size, hidden_size)
        V = (vocab_size, hidden_size)
        '''
        # Backprop
        loss_value, dU, dW, dV, db, dc, hidden_state = loss_func(x_seq, y_seq, hidden_state)
        derivatives = [dU, dW, dV, db, dc]
        parameters = [U, V, W, b, c]
        # Parameter update
        for parameter, derivative in zip(parameters, derivatives):
            parameter+= -1*learning_rate*derivative

        start_idx += sequence_length

    epoch += 1
