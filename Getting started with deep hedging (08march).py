#!/usr/bin/env python
# coding: utf-8

# In[64]:


# Author : Thibault
# Last updated : 08/03/2022

import sys, os
sys.path.insert(0, os.getcwd() + "/deep-hedging")
from IPython.display import clear_output
import numpy as np
import QuantLib as ql
import tensorflow as tf
from scipy.stats import norm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

clear_output()
print("\nAll libraries are now fully installed")


# In[46]:


# geometric Brownian motion
N = 30
s0 = 100.0
sigma = 0.2
r = 0.0
q = 0.0
k_train = 10**5
k_test_ratio = 0.2

# put option
K = s0
payoff_func = lambda x : -np.maximum(K-x, 0.0)

# proportional trading costs
eps = 0.0

# information set
info_set = "normalized_log_S"

# loss function
loss_type = "Entropy"
loss_param = 1.0

# neural network structure
m = 15 # neurons in each layer
d = 1 # hidden layers

# neural network training parameters
lr = 0.001
batch_size = 256
epochs = 50
use_batch_norm = False
kernel_initializer = "he_uniform"
activation_dense = "relu"
activation_output = "sigmoid"
final_period_cost = False
delta_constraint = (0.0, 1.0)
share_strategy_across_time = False
cost_structure = "proportional"

mc_simulator = "Numpy"


# In[45]:


seed = 0
nobs = int(k_train*(1+k_test_ratio))
dt = 1/N
matu = 1

dynamics = 0.0 # here we have to define our stochastic process (Black-Scholes in the paper)
spot = 0.0 # here we generate the path for the stochastic process (with given maturity, time steps and nobs)

clear_output()
print("\nSimulation done")


# In[55]:


# need to define our dynamics here
# function of initial spot, sigma, r, q, N, seed

# what this will return: 

def BlackScholesProcess(s0, sigma, r, q, N, seed):
    return


# In[54]:


# need to define our path generator here
# function of maturity, N, nobs

# what this will return: 

def PathGenerator(matu, N, nobs):
    return


# In[ ]:


payoff_final = payoff_func(spot[:,-1])
trading_set = np.stack((spot), axis=1)
I = np.stack((np.log(S)),axis=1)

# structure of x_train -> 1) trading set + 2) info set + 3) payoff
x_all = []
for i in range(N+1):
    x_all += [trading_set[i,:, None]]
    if i != N:
        x_all += [I[i,:, None]]
x_all += [payoff_final[:, None]]

# split entire data set into training + testing samples
test_size = int(k_train*k_test_ratio)
[x_train, x_test] = train_test_split(test_size = test_size)
[S_train, S_test] = train_test_split(test_size = test_size)
[payoff_final_train, payoff_final_test] = train_test_split([x_all[-1]], test_size = test_size)

print("\nFinished preparing data")


# In[ ]:


# setup and model compilation
optimizer = Adam(learning_rate = lr)
simpleModel = 0.0 # need to define the big deep hedging model
loss = 0.0 # need to define the loss function
simpleModel.add_loss(loss)
simpleModel.compile(optimizer=optimizer)

# fit the model
simpleModel.fit(x = x_train, batch_size = batch_size, epochs = epochs, validation_data = x_test, verbose = 1)
clear_output()
print("\nFinished running the deep hedging algorithm for a simple network")


# In[ ]:


# setup and model compilation
optimizer = Adam(learning_rate = lr)
recurrentModel = 0.0 # need to define the big deep hedging model
loss = 0.0 # need to define the loss function
recurrentModel.add_loss(loss)
recurrentModel.compile(optimizer=optimizer)

# fit the model
recurrentModel.fit(x = x_train, batch_size = batch_size, epochs = epochs, validation_data = x_test, verbose = 1)
clear_output()
print("\nFinished running the deep hedging algorithm for a recurrent network")


# In[57]:


# need to define the loss function here (preferably the expected shortfall)

# what this will return :

def expectedShortfall():
    


# In[51]:


# need to define here the deep hedging model (can be either simple or recurrent)

# what this will return :

def deepHedging():
    


# In[ ]:


put = 0.0 # need to define a function for the european put
put_price = 0.0 # need to define the black-scholes price of a put
put_delta = 0.0 # need to define the black-scholes delta of a put
put_pnl = 0.0 # need to define the corresponding pnl of the whole operation

risk_neutral_price = -payoff_final_test[0].mean()*np.exp(-r*(N*dt))
nn_simple_price = simpleModel.evaluate(x_test, batch_size = test_size, verbose = 0)
print(risk_neutral_price)
print(nn_simple_price)

try:
    nn_recurrent_price = recurrentModel.evaluate(x_test, batch_size = test_size, verbose = 0)
    print(nn_recurrent_price)
except:
    print("no recurrent model")


# In[56]:


# need to define properly the european put option

# what this will return :

class putOption():
    


# In[ ]:


# need to define the black-scholes put price

# what this will return :

def putPrice():
    


# In[ ]:


# need to define the black-scholes put delta

# what this will return :

def putDelta():
    


# In[ ]:


# need to define the total corresponding pnl

# what this will return :
def PnL():
    


# In[61]:


# --------------------------------------------------
# what we did today - the questions we still have not got any answer to - what to do
# --------------------------------------------------

# we kinda got our hands dirty and dug into the complex code used to deep hedge derivatives
# we kinda got the general idea of the structure
# even though we did not understand most of what was going on

# we really want not to use what has already been implemented
# we really want to create something of our own (fully)
# because it is more fun, because I dont want to plagiarize, because I will work on rainbow options

# so we so far copied the general structure for the program (without having really coded the algo)
# we understood everything that we wrote so far
# we need to figure out how to build the paths and compute the deltas
# we also need to figure out how to implement them, beyond the actual formula
# need to be familiar with classes, so that we can define our own classes for vanillas and rainbows

# we will still work with the following
# vanillas: black-scholes and heston paths, black-scholes hedge
# rainbows: multi-asset geometric brownian motion and kou jump-diffusion, stulz hedge

# we also need to build the functions for the loss functions
# we also need to build the functions for the pnl (and understand the meaning behind)

