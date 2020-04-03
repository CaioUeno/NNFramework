#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from hyper_parameters import *
from layers import *


# In[2]:


class network(object):
    
    ''' class which represents a neural network '''
    
    def __init__(self, task):
        
        self.task = task # avaliable values = ['classification', 'regression']
        self.layers = []
        self.loss = mse_loss()
        self. history_loss = None
        
    def add(self, layers):
        
        ''' Add a layer (or a list of layers) to the model '''
        
        self.check_layer(layers)
        
        if self.layers: # if there's at least a layer, then change its hidden atribute
            self.layers[-1].hidden = True
        
        if isinstance(layers, neutral_layer): # if layers is actually only one layer
            layers.hidden = False
            self.layers.append(layers)
        
        else: # pass a list of layers
            layers[-1].hidden = False
            self.layers += layers
        
    def predict_one_sample(self, sample):
        
        ''' Pass the sample over the network '''
        
        for idx, layer in enumerate(self.layers):
            
            if idx == 0:
                outs = layer.feed_forward(sample)
            
            else:
                outs = layer.feed_forward(outs)
                
        return outs
    
    def backpropagation(self, samples, targets):
        
        losses = [] # save losses over each sample
        
        for sample, target in zip(samples, targets):
            self.predict_one_sample(sample) 

            # LAYERS' ORDER REVERSED TO CALCULATE DELTAS
            self.layers.reverse()
            for idx, layer in enumerate(self.layers):
                
                if len(self.layers) == 1: # if network has only one layer
                    layer.calculate_delta(sample, loss_derivative_applied=self.loss.derivative(target, layer.out))

                elif idx == 0: # last layer (layers' list is reversed)
                    layer.calculate_delta(self.layers[idx+1].out, loss_derivative_applied=self.loss.derivative(target, layer.out))

                elif idx == len(self.layers)-1: # first layer (layers' list is reversed)
                    layer.calculate_delta(sample, next_layer=self.layers[idx-1])

                else: # hidden layers between first and last layer
                    layer.calculate_delta(self.layers[idx+1].out, next_layer=self.layers[idx-1])
                    
                losses.append(sum(self.loss.apply(target, self.layers[0].out)))

            #LAYERS' ORDER RECOVERED
            self.layers.reverse()
        
        for layer in self.layers:
            layer.update_weights(len(samples))
                   
        return np.mean(losses)
            
    def train(self, X, y, batch_size=32, epochs=20):
        
        self.check_X_y(X, y)
        
        times_batch = len(X) / batch_size # how many batches will iterate
        
        if not times_batch.is_integer(): # add one if the division is not exact
            times_batch = times_batch + 1
        
        times_batch = int(times_batch)
        
        history_losses = []
        
        for epoch in range(epochs):
            
            epoch_loss = [] # save epoch loss
            for t_batch in range(times_batch):
                batch_loss = self.backpropagation(X[t_batch * batch_size:(t_batch+1) * batch_size],
                                                  y[t_batch * batch_size:(t_batch+1) * batch_size])
                epoch_loss.append(batch_loss)
            
            history_losses.append(np.mean(epoch_loss))
            
        self.history_loss = history_losses
            
    def predict(self, X, raw=True): # raw means you want the scores of last neurons
        
        self.check_X(X)
        
        preds = []
        for x in X:
            pred = self.predict_one_sample(x)
            
            if not raw and self.task == 'classification':
                preds.append(np.array(pred).argmax())
            
            else:
                preds.append(pred)
                
        return np.array(preds)
    
    def evaluate(self, X, y):
        
        if self.task == 'classification':
            preds = self.predict(X, raw=False)
            return sum(y_test.argmax(axis=1) == preds) / len(preds)
        
        else:
            return np.mean(self.loss.apply(y, self.predict(X)))
        
    def check_layer(self, layers):
        
        check_types = isinstance(layers, neutral_layer) or                       all([isinstance(layer, neutral_layer) for layer in layers])
        
        if not check_types:
            raise TypeError('must pass neutral_layer obj(s)')
        
        if not self.layers and isinstance(layers, neutral_layer): # if add one layer to a empty network
            
            check_sizes = True
            
        elif not self.layers and all([isinstance(layer, neutral_layer) for layer in layers]): # if add layers list to
                                                                                              # empty network
            
            if not all([layers[i].n_neurons == layers[i+1].n_input for i in range(len(layers)-1)]):
                raise ValueError('neurons\' sizes over network don\'t match.')
            
        elif self.layers and isinstance(layers, neutral_layer): # if add one layer to a NOT empty network
            
            if not self.layers[-1].n_neurons == layers.n_input:
                raise ValueError('neurons\' sizes over network don\'t match.')
            
        elif self.layers and all([isinstance(layer, neutral_layer) for layer in layers]): # if add layers list to
                                                                                          # a NOT empty network
            if not (self.layers[-1].n_neurons == layers[0].n_input and                    all([layers[i].n_neurons == layers[i+1].n_input for i in range(len(layers)-1)])):
                raise ValueError('neurons\' sizes over network don\'t match.')
            
        
    def check_X_y(self, X, y):
        
        ''' Check inputs '''
        
        if len(X) != len(y):
            raise ValueError('X and y doesn\'t have same length. X: '+ str(len(X)) + ', y:' + str(len(y)))
            
        self.check_X(X)
        self.check_y(y)
 
    def check_X(self, X):
        
        if len(X.shape) != 2:
            raise ValueError('X samples don\'t have the same length.')
        
        if X.shape[1] != self.layers[0].n_input:
            raise ValueError('X samples don\'t have same shape as first layer\'s input.')
        
    def check_y(self, y):
        
        if len(y.shape) != 2:
            raise ValueError('y targets don\'t have the same length.')
        
        if y.shape[1] != self.layers[-1].n_neurons:
            raise ValueError('target (y) doesn\'t have same shape as last layer.')
        

