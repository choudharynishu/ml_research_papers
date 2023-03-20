'''
This is the basic introduction to PyTorch package
Tutorial material is obtained from the official website
https://pytorch.org/tutorials/
Authors of the Tutorials:
Soumith Chintala, Suraj Subramanian, Seth Juarez, Cassie Breviu, Dmitry Soshnikov, and Ari Bornstein

06/16/2022
'''
#-----Installation-----#
#Pip installation command for local installation on MacOS (Type this in Local)
#pip3 install torch torchvision torchaudio

import torch

'''
Two Primitives to work with data in PyTorch
1. Dataset, which contains the observations as inputs and target labels and, 
2. DataLoader, "which wraps an iterable around Dataset" - so basically can iterate through the data 

Additionally, PyTorch has domain specific packages such as TorchText, TorchAudio, and TorchVision
For this tutorial authors have used the FashionMNIST dataset. 

According to the Docs every dataset contains following keyword arguments
   a. root: str type, is the path where the train/test data is stored,
   b. loader (callable): A function to load a sample given its path
   c. extensions (tuple[string]) – A list of allowed extensions. both extensions and is_valid_file should not be passed
   d. transform (callable, optional) – A function/transform that takes in a sample and returns a transformed version.
   e. target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
   f. is_valid_file – Checked only when extentions are passed. A function that takes path of a file and 
                      check if the file is a valid file
'''

#-----Import commands-----#
from torch import nn #most likely to construct simple feed-forward networks
from torchvision import datasets #from here we are going to extract the FashionMNIST data
from torch.utils.data import DataLoader #primitive to work with data
from torchvision.transforms import ToTensor

#-----Extracting the FashionMNIST dataset into training and test set-----#
#This will only download once
training_data = datasets.FashionMNIST(root="data",
                                      train=True, #specifies training or test dataset,
                                      download=True, #downloads the data from the internet if it’s not available at root.
                                      transform=ToTensor() #specify the feature transformations
                                      )
testing_data = datasets.FashionMNIST(root="data",
                                     train=False,
                                     download=True,
                                     transform=ToTensor())
#-----Create an iterator out of the downloaded data-----#
batch_size = 64

#Create dataloaders

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)

#Demonstration of what DataLoader does
for X, y in test_dataloader:
    #Here N= Batch-size- specified number of observations
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(y)
    break

#-----Define the Neural Network model-----#
#Machine to use for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} machine")

#For making the Neural Network - we define a class that inherits from nn.Module class
'''
Useful tidbits about class inheritence 
class C(B):
    def method(self, arg):
        super().method(arg)    # This does the same thing as: super(C, self).method(arg)
'''

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                                              #Linear module applies a linear transformation on the input
                                              #using its stored weights and biases.
                                              #This layer receives input from the previous flatten layer
                                              nn.Linear(in_features=28*28, out_features=512),
                                              nn.ReLU(),
                                              nn.Linear(in_features=512, out_features=512),
                                              nn.ReLU(),
                                              nn.Linear(in_features=512, out_features=10)
                                            )
    #Define forward pass on one particular image
    def forward(self, x):
        #print(x.size())
        x = self.flatten(x) #this basically calls nn.Flatten(x)
        #print(f"Flattened size, {x.size()}")
        logits = self.linear_relu_stack(x) #this calls the architecture defined in this attributes
        return logits

model = NeuralNetwork().to(device)
#print(dir(model))

#-----Training the Neural Network model-----#
loss_function = nn.CrossEntropyLoss()                      #Loss-function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)   #Optimization function

def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        #Forward pass
        # Compute prediction error and loss value
        pred = model(X)
        #print(f"predict is {pred}")
        loss = loss_function(pred, y)
        #Backpropagation
        #Optimizer.zero_grad(): sets the gradient of all optimized Torch.tensors to zero
        optimizer.zero_grad()
        loss.backward()
        #Optimizer.step() performs a single step(parameter update)
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return None

def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_of_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    #  Turn off the grad - mainly for efficiency as we won't be updating any parameters
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            pred = model(X)
            print(f"Trained model's predicted tensor's size, {pred.size()}")
            print(f"Actual y, {y}")
            test_loss += loss_function(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_of_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return None

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_function, optimizer)
    test(test_dataloader, model, loss_function)

print("Done!")
