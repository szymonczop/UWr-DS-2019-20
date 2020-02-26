#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:47:38 2019

@author: czoppson
"""

from torch import nn

"""
PyTorch provides a module nn that makes building networks much simpler. 
We’ll see how to build a neural network with 784 inputs, 
256 hidden units, 10 output units and a softmax output. """

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256) # from initial to hidden_1
        
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10) # from hidden 1 to output
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x): ## tutaj tak naprawdę definiuje po koleji co i jak idzie 
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x


model = Network()
model

input_size = 784
hidden_sizes = [128, 64]
output_size = 10# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

model[0]





class Net(nn.Module):        
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                   nn.Linear(hidden_dim, output_dim),
                                   #nn.Sigmoid())
                                    nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        x = self.layers(x)
        return x
    
    
stypek = Net(786,128,64)
stypek
