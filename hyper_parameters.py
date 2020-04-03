#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from scipy.special import expit


# # Activation fuctions

# In[ ]:


class f_activation(object):
    ''' Model of a activation function's structure. '''
    
    def apply(self, x):
        return None
    
    def derivative(self, x):
        return None


# In[ ]:


class identity_activation(f_activation):
    
    def __init__(self):
        
        self.type_ = 'tanh'
        
    def apply(self, x):
        
        return x
    
    def derivative(self, x):
        
        return np.ones(len(x))


# In[ ]:


class tanh_activation(f_activation):
    
    def __init__(self):
        
        self.type_ = 'tanh'
        
    def apply(self, x):
        
        return np.tanh(x)
    
    def derivative(self, x):
        
        return 1 - (np.tanh(x) ** 2)


# In[ ]:


class sigmoid_activation(f_activation):
    
    def __init__(self):
        
        self.type_ = 'sigmoid'
        
    def apply(self, x):
        
        return expit(x)
    
    def derivative(self, x):
        
        return expit(x) * (1 - expit(x))


# In[ ]:


class sin_activation(f_activation):
    
    def __init__(self):
        
        self.type_ = 'sin'
        
    def apply(self, x):
        
        return np.sin(x)
    
    def derivative(self, x):
        
        return np.cos(x)


# # Loss functions

# In[ ]:


class loss_function(object):
    
    ''' Model of a loss function's structure. '''
    
    def apply(self, true, pred):
        
        return None
        
    def derivative(self, true, pred):
        
        return None


# In[ ]:


class mse_loss(object):
    
    def __init__(self):
        
        self.type_ = 'mse'
        
    def apply(self, true, pred):
        
        return 0.5 * ((true - pred) ** 2)
        
    def derivative(self, true, pred):
        
        return pred - true        


# In[37]:


class binary_loss(object):
    
    def __init__(self):
        
        self.type_ = 'binary'
        
    def apply(self, true, pred):

        return - np.nan_to_num(true * np.nan_to_num(np.log(pred)) + (1 - true) * np.nan_to_num(np.log(1 - pred)))
        
    def derivative(self, true, pred):
        
        dv = np.zeros(len(true))
        
        for i in range(len(true)):
            
            if true[i] == 1:
                dv[i] = - true[i] / pred[i]
                
            else:
                dv[i] = - (1 - true[i]) / (1 - pred[i])
                
        return np.nan_to_num(dv)

