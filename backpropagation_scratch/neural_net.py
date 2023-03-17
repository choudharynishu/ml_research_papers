import math
import random
from backprop import value

class Neuron:
    def __init__(self, n_inputs):
        #weights initialized using uniform random distribution
        self.w = [value(random.uniform(-1, 1)) for i in range(n_inputs)]
        #single bias
        self.b = value(random.uniform(-1, 1))
    def __call__(self, x):
        #assert dimension compatible for dot product
        output = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        activation = output.relu() #later, replace with Relu
        return activation
    def parameters(self):
        return self.w +[self.b]

class Layer:
    def __init__(self, n_outputs, n_inputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output[0] if len(output)==1 else output
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, n_input, hidden_layers, n_output):
        size = [n_input] + hidden_layers + [n_output]
        self.layers = [Layer(size[i+1], size[i]) for i in range(len(size)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [lp for layer in self.layers for lp in layer.parameters()]
x = [[2.0, 3.0, 1.0],
     [0.5, -1.0, 0.5],
     [3.0, -2.0, 1.1],
     [1.0, 1.0, 1.0]]
y = [1.0, -1.0, -2.0, 1.0]
'''
Neural network with input layer of shape = (, 3)
Two hidden layers with 4 nodes each
Finally, the output layer
Activation function is Relu, later modify to accept str input for activation
'''
neural_net = MLP(3, [4, 4], 1)
ypred = [neural_net(xi) for xi in x]
print(ypred)
loss = sum((ypredi-yi)**2 for ypredi, yi in zip(ypred, y))#Mean-squared loss

learning_rate = 0.01
max_iter = 20
tol = 0.1

iteration = 0
while math.sqrt(loss.data) >= tol and iteration < max_iter:
    loss.backward()
    #print(len(neural_net.parameters()))
    for p in neural_net.parameters():
        p.data += (-1.0*learning_rate*p.grad)
    ypred = [neural_net(xi) for xi in x]
    #print(ypred)
    loss = sum((ypredi - yi) ** 2 for ypredi, yi in zip(ypred, y))
    for p in neural_net.parameters():
        p.grad = 0.0
    iteration += 1
print(iteration)
ypred = [neural_net(xi) for xi in x]
print(ypred)
