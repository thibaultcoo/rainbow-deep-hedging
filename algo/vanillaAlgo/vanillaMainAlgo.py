#!/usr/bin/env python
# coding: utf-8

# Author : Thibault
# Last updated : 21/03/2022

#--------------------------------------------------------------
# deep hedging model for vanilla options

#---------------------- loading packages ----------------------
import sys

sys.path.append("")
from IPython.display import clear_output
import numpy as np
from tensorflow.compat.v1.keras.optimizers import Adam
from scipy.stats import norm
import matplotlib.pyplot as plt

from dynamics.singleGeometricBrownian import GeometricBrownianMotion
from functions.losses import Entropy, CVaR, MSE
from functions.train_test_split import train_test_split
from instruments.EuropeanPut import EuropeanPut

from algo.vanillaAlgo.vanillaHedging import vanillaHedging

clear_output()
print("\nAll libraries are now fully installed")
#--------------------------------------------------------------

#---------------- initializing constants ----------------------
# geometric Brownian motion
N = 30
s0 = 100.0
sigma = 0.2
r = 0.0
k_train = 1*(10**5)
k_test_ratio = 0.2
K = s0
eps = 0.0

payoffFunction = lambda x : -np.maximum(K-x, 0.0) # payoff function of a put option

info_set = "normalized_log_S"
loss_param = 1.0 # we will use the entropy risk measure (convex risk measure)

# neural network structure
m = 15 # neurons in each layer
d = 1 # hidden layers
lr = 0.001 # learning rate
batch_size = 256 # slices of the data set that the algo is trained on
epochs = 50 # number of times the algo will loop through the whole set
use_batch_norm = False
kernel_initializer = "he_uniform" # initialization of weights and biases
activation_dense = "leakyReLU" # activation function for the hidden layers
activation_output = "sigmoid" # activation function for the output layer
final_period_cost = False
share_strategy_across_time = False

seed = 0 # to randomize
nobs = int(k_train*(1+k_test_ratio)) # number of total observations
matu = 1 # maturity of our option
#--------------------------------------------------------------

#---------- loading dynamics and preparing the date -----------
dynamics = GeometricBrownianMotion(s0 = s0, sigma = sigma, r = r, matu = matu, N = N)
spot = dynamics.gen_path(nbPaths = nobs)

print(np.shape(spot))

clear_output()
print("\nSimulation done")

payoff_final = payoffFunction(spot[:,-1]) # vector of final payoffs for all nobs
trading_set = np.stack((spot), axis=1) # set of underlying values for all spots
I = np.stack((np.log(spot/s0)),axis=1) # set of normalized-log transform of all those values

# structure of x_train -> 1) trading set + 2) info set + 3) payoff
x_all = []
for i in range(N+1):
    x_all += [trading_set[i,:,None]]
    if i != N:
        x_all += [I[i,:,None]]
x_all += [payoff_final[:,None]] # we append the final payoffs to our input total set

# split entire data set into training + testing samples
test_size = int(k_train*k_test_ratio)
[x_train, x_test] = train_test_split(x_all, test_size = test_size)
[S_train, S_test] = train_test_split([spot], test_size = test_size)
[payoff_final_train, payoff_final_test] = train_test_split([x_all[-1]], test_size = test_size)

print(x_test)

print("\nFinished preparing data") # the data is well split between the two
#--------------------------------------------------------------

#-------------- setup and model compilation -------------------
optimizer = Adam(learning_rate = lr)
simpleModel = vanillaHedging(N=N, d=d+2, m=m, r=r, dt=matu / N, init_w=0, eps=eps, T_cost=False, \
                             strat_type="simple", use_batch_norm=use_batch_norm, \
                             kernel_initializer=kernel_initializer, \
                             activation_dense=activation_dense, \
                             activation_output=activation_output, share_strat_across_time=False
                             )

#loss = CVaR(w=simpleModel.output, wealth=0, loss_param=0.5)
loss = MSE(w = simpleModel.output)
#loss = Entropy(w=simpleModel.output, loss_param=loss_param)
simpleModel.add_loss(loss)
simpleModel.compile(optimizer=optimizer)

# fit the model
simpleModel.fit(x = x_train, batch_size = batch_size, epochs = epochs, validation_data = x_test, verbose = 1)
clear_output()
print("\nFinished running the deep hedging algorithm for a simple network")
#--------------------------------------------------------------

#-------------------- benchmark comparison --------------------
put = EuropeanPut()
put_price = put.get_BS_price(S=S_test[0], sigma=sigma, r=r, K=K, matu=matu, N=N)
put_delta = put.get_BS_delta(S=S_test[0], sigma=sigma, r=r, K=K, matu=matu, N=N)
put_pnl = put.get_BS_PnL(S=S_test[0], payoff=payoffFunction(S_test[0][:,-1]), delta=put_delta, matu=matu, r=r, \
                         final_period_cost=True, eps=eps, N=N)

risk_neutral_price = -payoff_final_test[0].mean()*np.exp(r * matu)
nn_simple_price = simpleModel.evaluate(x_test, batch_size = test_size, verbose = 0)

print("The Black-Scholes model price is %2.3f." % put_price[0][0])
print("The Risk Neutral price is %2.3f." % risk_neutral_price)
print("The Deep Hedging price is %2.3f." % nn_simple_price)
#--------------------------------------------------------------

#----------------------- building graphs ----------------------
benchmarkBars = put_pnl + put_price[0][0]
algoBars = simpleModel(x_test).numpy().squeeze() + put_price[0][0]
PnL_figure = plt.figure(dpi=125, facecolor='w')
ax = PnL_figure.add_subplot()
ax.hist((benchmarkBars,algoBars), bins=30, label=['Black-Scholes PnL', 'Deep Hedging PnL'])
ax.legend
plt.show()
#--------------------------------------------------------------

