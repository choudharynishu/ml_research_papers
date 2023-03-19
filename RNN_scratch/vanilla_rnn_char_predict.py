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
sequence_length = 50  # No of x-vectors to estimate the final hidden state
hidden_size = 100  # Number of neurons in the hidden layer of RNN (only one used here)
learning_rate = 0.01  # Standard learning-rate

# Initialize the parameters of the model
# Multiplied by 0.01 for smaller (better) initializations
U = np.random.randn(hidden_size, vocab_size) * 0.01
W = np.random.randn(hidden_size, hidden_size) * 0.01
V = np.random.randn(vocab_size, hidden_size) * 0.01
b = np.zeros((hidden_size, 1))  # Bias applied to the hidden state (before activation)
c = np.zeros((vocab_size, 1))  # Bias applied before the output vector (before activation)
parameters = [U, W, V, b, c]
# General user inputs
total_epochs = 5
epoch, start_idx = 0, 0  # n = current number of iteration


# Define the loss function and update parameters
def loss_func(x_seq, y_seq, hidden_s):
    '''
        :param x_seq: input x-sequence containing a character's index in our language dictionary
        :param y_seq: index of next character in the sequence
        :param hidden_s: estimate of hidden state (parameters)
        :return: Value of loss function, derivatives w.r.t all parameters, and hidden state
    '''
    # Forward-propagation
    # These dictionaries will contain one-hot encoding representation of input, output, and probabilities
    # Also the hidden state is kept track to estimate gradient through backpropagation in time
    assert len(x_seq) == len(y_seq), "Input and Output sequences should be of the same length"

    x_dict, h_dict, y_dict, prob_dict = {}, {}, {}, {}
    loss_value = 0
    h_dict[-1] = np.copy(hidden_s)

    for t in range(len(x_seq)):
        x_dict[t] = np.zeros((vocab_size, 1))
        x_dict[t][x_seq[t]] = 1.0  # assigning probability = 1 for index
        h_dict[t] = np.tanh(np.dot(U, x_dict[t])+np.dot(W, h_dict[t-1])+b)  # Next Hidden state
        y_dict[t] = np.dot(V, h_dict[t])+c # Un-normalized probits (? recheck)
        prob_dict[t] = np.exp(y_dict[t])/np.sum(np.exp(y_dict[t]))  # Passing prediction through Softmax layer/unit
        loss_value += -np.log(prob_dict[t][y_seq[t], 0]) # take the probability for the index of true prediction

    # Backpropagation Through Time (BPTT)
    dU, dW, dV, db, dc = map(lambda x: np.zeros_like(x), parameters)
    dh_tplus = np.zeros_like(h_dict[0])
    for t in reversed(range(len(x_seq))):
        d_prob = np.copy(prob_dict[t])
        d_prob[y_seq[t]] -= 1.0
        dV += np.dot(d_prob, h_dict[t].T)
        dc += d_prob
        dh = np.dot(V.T, d_prob)+ dh_tplus
        dh_tplus_raw = (1 - h_dict[t] * h_dict[t]) * dh
        db += dh_tplus_raw
        dW += np.dot(dh_tplus_raw, h_dict[t-1].T)
        dU += np.dot(dh_tplus_raw, x_dict[t].T)
        dh_tplus = np.dot(W.T, dh_tplus_raw)

    for dparam in [dU, dW, dV, db, dc]:
        np.clip(dparam, -5, 5, out=dparam)
    return loss_value, dU, dW, dV, db, dc, h_dict[len(x_seq)-1]


while epoch <= total_epochs:
    # After every epoch reset the RNN memory
    start_idx = 0
    hidden_state = np.zeros((hidden_size, 1))
    while start_idx + sequence_length + 1 < len(data):
        x_seq = [char_to_index[char] for char in data[start_idx:start_idx + sequence_length]]
        y_seq = [char_to_index[char] for char in data[start_idx + 1:start_idx + sequence_length + 1]]
        #if epoch % 5 == 0:
            #Print current value of loss
        # Backprop
        loss_value, dU, dW, dV, db, dc, hidden_state = loss_func(x_seq, y_seq, hidden_state)
        #print(f"Loss value: {loss_value}")
        derivatives = [dU, dW, dV, db, dc]
        parameters = [U, W, V, b, c]
        # Parameter update
        for parameter, derivative in zip(parameters, derivatives):
            parameter += -1 * learning_rate * derivative
        start_idx += sequence_length
    print(f"Loss value: {loss_value}")
    epoch += 1
print(f"Loss value: {loss_value}")