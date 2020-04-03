#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from copy import deepcopy

from hyper_parameters import *


# In[2]:


class neutral_layer(object):
    
    ''' Model of a layer's structure '''
    
    def __init__(self, n_input, n_neurons, activation, learning_rate, hidden):
        
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.hidden = hidden
        
        self.weights = np.random.random((n_neurons, n_input)) 
        self.bias = np.random.random((n_neurons))
        
        self.choose_activation(activation)

        self.out = None
        self.delta = None
        self.update = np.zeros((n_neurons, n_input))
        self.learning_rate = learning_rate
        
    def feed_forward(self, in_features):
        
        ''' Function apllied at in_features '''
        
        return None
        
    def choose_activation(self, activation): 
        
        if activation == 'tanh':
            self.activation = tanh_activation()
        
        elif activation == 'sigmoid':
            self.activation = sigmoid_activation()
            
        elif activation == 'sin':
            self.activation = sin_activation()
            
        elif activation == 'identity':
            self.activation = identity_activation()
            
    def derivative_in(self, in_features):
        
        ''' Feed foward function's derivative with respect to in_features '''
        
        return None
    
    def derivative_weights(self, in_features):
        
        ''' Feed foward function's derivative with respect to self weights '''
        
        return None
    
    def calculate_delta(self, in_features, loss_derivative_applied=None, next_layer=None):
        
        ''' Calculate layer's delta and layer weights' update values '''
        
        if not self.hidden:
            self.delta = loss_derivative_applied * (self.activation.derivative(self.out))
    
        else: # if last layer only
            self.delta = (next_layer.delta @ next_layer.derivative_in(self.out)) * (self.activation.derivative(self.out)) 

        self.update += (self.delta * self.derivative_weights(in_features).T).T
        
    def update_weights(self, batch_size):
        
        ''' Update weights using mini-batch strategy '''
        
        self.update = (1/batch_size) * self.update
        self.weights -= self.learning_rate * self.update
        self.update = np.zeros((self.n_neurons, self.n_input))
        
    def copy(self):
        
        return deepcopy(self)


# In[ ]:


class linear_layer(neutral_layer):
    
    def __init__(self, n_input, n_neurons, activation='tanh', learning_rate=0.5, hidden=True):
        
        super().__init__(n_input, n_neurons, activation, learning_rate, hidden)
        
    def feed_forward(self, in_features):
        
        self.out = self.activation.apply(self.weights @ in_features + self.bias)
        return self.out
    
    def derivative_in(self, in_features):
        
        return self.weights
    
    def derivative_weights(self, in_features):
        
        return np.array([in_features] * self.n_neurons)


# In[ ]:


class senoidal_layer(neutral_layer):
    
    def __init__(self, n_input, n_neurons, activation='tanh', learning_rate=0.5, hidden=True):
        
        super().__init__(n_input, n_neurons, activation, learning_rate, hidden)
        
    def feed_forward(self, in_features):
        
        self.out = self.activation.apply(np.sin(self.weights @ in_features) + self.bias)
        return self.out
    
    def derivative_in(self, in_features):
        
        return np.cos(self.weights * np.array([in_features] * next_layer.n_neurons)) * self.weights
    
    def derivative_weights(self, in_features):
        
        return np.cos(self.weights * np.array([in_features] * self.n_neurons)) * np.array([in_features] * self.n_neurons)

