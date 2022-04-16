#--------------------- importing the required libraries -------------
from IPython.display import clear_output # cleaner outputs
import numpy as np # various calculations

from tensorflow.compat.v1.keras.optimizers import Adam # stochastic gradient descent algorithm used (see chap3)
import matplotlib.pyplot as plt # nice looking graphs

from options.EuropeanWorstofTwoCall import EuropeanWorstofTwoCall # class embedded with functions for its price, pnl and delta
from rainbowAlgo.rainbow_generalAlgo import rainbowModel # class embedded with the general algorithm for the rainbow hedging
from functions.losses import Entropy, MSE # loss functions
from functions.splitRainbow import set_split_rainbow # will split our entire data sets into a training and testing set

from dynamics.rainbowDyn.multiGeomBrownian import multiGeometric # used to generate regular geometric Brownian motion paths
from dynamics.rainbowDyn.kouJumpDiffusion import multiJumpDiffusion # used to generate multi-asset paths with Kou Jump Diffusion
#---------------------------------------------------------------------
clear_output()
print("\nAll libraries are now imported.")
#------------------- initializing our constants ----------------------
N = 30 # number of time steps
T = 1 # maturity
S0 = (100.0, 100.0) # initial spot values
m = (0.1, 0.1) # rates of returns for both our underlyings
cov = np.zeros((2,2)) # initializing the covariance matrix for both our returns
cov[0,0] = 0.05 # variance of underlying 1
cov[1,1] = 0.05 # variance of underlying 2
cov[0,1] = 0.03 # covariance between 1 and 2
cov[1,0] = 0.03 # covariance between 2 and 1
r = 0.0 # risk-free rate

Ktrain = 1*(10**3) # date points in the training set
Ktest_ratio = 0.2 # data points in the testing set as a share of the training set

strike = np.mean(S0) # strike level of the rainbow option
payoff_rainbow = lambda x, y: -np.maximum(0, np.minimum(x-strike, y-strike)) # payoff of the rainbow option (see references)

eps = 0.0 # proportional transaction costs
nb_neurons = 25 # number of neurons per layer
nb_hidden = 2 # number of hidden layers

lr = 0.005 # learning rate
batch_size = 500 # batch size
epochs = 10 # number of epochs
kernel_initializer = "he_uniform"

activation_dense = "relu" # activation function applied to the output of neurons on hidden layers
activation_output = "sigmoid" # activation function applied to the output of neurons on the output layer

seed = 0 # used to randomize
nobs = int(Ktrain*(1+Ktest_ratio)) # total number of observations (sum of training and testing sets)

dt = 1/365 # we consider daily rebalancing: dt is the amount of a year that goes by in a day

# Kou jump-diffusion extension parameters
prob1 = 0.20 # ratio of the Poisson parameter for asset 1 dynamics, and the total Poisson parameter (see explanation in chap5)
prob2 = 0.20 # ratio of the Poisson parameter for asset 2 dynamics, and the total Poisson parameter (see explanation in chap5)
#------------------------- generating our price paths ----------------
#process_Geom = multiGeometric(s0=S0, T=T, N=N, cov=cov, dt=dt)
process_Kou = multiJumpDiffusion(m=m, s0=S0, T=T, N=N, cov=cov, prob1=prob1, prob2=prob2, dt=dt)

#spot_Geom = process_Geom.gen_path(nobs)
spot_Geom = process_Kou.gen_path(nobs)
#---------------------------------------------------------------------
clear_output()
print("\nAll price paths are now created.")
#------------------------- preparing the data sets -------------------
finalPayoff = payoff_rainbow(spot_Geom[:, -1, 0], spot_Geom[:, -1, 1]) # computes the payoffs at the final date of all paths
tradingSet = np.stack((spot_Geom), axis=1) # stacks all layers comprised of the spot paths
infoSet = np.stack((np.log(spot_Geom / strike)), axis=1) # stacks those same layers, log-transformed and centered on the strike

x_all = [] # initializing the matrix that will comprise the entire dataset
for i in range(N+1):
    for j in range(2): # we will successively stack the layers in that order: prc_1, info_1, prc_2, info_2
        x_all += [tradingSet[i, :, j, None]]
        if i != N:
            x_all += [infoSet[i, :, j, None]]
x_all += [finalPayoff[:, None]] # we end up by adding the final payoffs layer

test_size = int(Ktrain*Ktest_ratio) # we denote as such the size of the testing set

# we split the data into our two sets
[x_train, x_test] = set_split_rainbow(x_all, test_size=test_size)
[s_train, S_test] = set_split_rainbow([spot_Geom], test_size=test_size)
[payoff_train, payoff_test] = set_split_rainbow([x_all[-1]], test_size=test_size)
#----------------------------------------------------------------------
print("\nAll data sets are now prepared.")
#--------------------- running the algorithm --------------------------
optimizer = Adam(learning_rate=lr) # initializing the gradient descent algorithm with the chosen learning rate

doubleModel = rainbowModel(N=N, nb_neurons=nb_neurons, nb_hidden=nb_hidden+2, r=r, dt=dt, eps=eps,
                           kernel_initializer = kernel_initializer, activation_dense = activation_dense,
                           activation_output = activation_output) # initializing the entire model

# calls either one of the two loss functions (disclaimer: using Entropy for Jump-diffusion paths tends to output NaNs)
#loss = Entropy(doubleModel.output, loss_param=1.0)
loss = MSE(doubleModel.output, nobs)
doubleModel.add_loss(loss)

doubleModel.compile(optimizer=optimizer) # compiling the model

# we run the neural networks under the afordefined stochastic paths
doubleModel.fit(x=x_train, batch_size=batch_size, epochs=epochs, validation_data=x_test, verbose=1)
#----------------------------------------------------------------------
clear_output()
print("\nThe algorithm has now finished running.")
print("\nThe Deep Hedging strategy, as well as the predicted price of the rainbow option, will be shown soon.")
#---------------------- benchmark comparison --------------------------
option = EuropeanWorstofTwoCall()

price_Stulz = option.get_Stulz_price(S=S_test[0], cov=cov, r=r, K=strike, matu=T,
                                     N=N, nbPaths=1, dt=dt) # we compute the price of the benchmark (Stulz)

delta_Stulz = option.get_Stulz_delta(S=S_test[0], cov=cov, r=r, K=strike, matu=T,
                                     N=N, nbPaths=1, dt=dt) # we compute the deltas of the benchmark (Stulz)

PnL_Stulz = option.get_Stulz_PnL(S=S_test[0],
                                 payoff=payoff_rainbow(S_test[0][:, -1, 0], S_test[0][:, -1, 1]),
                                 delta=delta_Stulz, matu=T, r=r, eps=eps, N=N, dt=dt) # we compute the overall benchmark PnL (Stulz)

mc_price = -payoff_test[0].mean() * np.exp(-r * (N * dt)) # we compute the price using a classic computation method (Monte-Carlo)
algo_price = doubleModel.evaluate(x_test, batch_size=test_size, verbose=0) # we output what the algorithm produces as a price
#----------------------------------------------------------------------
print("The Stulz model price is %2.3f." % price_Stulz[0][0])
print("The Monte-Carlo price is %2.3f." % mc_price)
print("The Deep Hedging price is %2.3f." % algo_price)
#------------------- building the comparison graph --------------------
bar1 = PnL_Stulz + price_Stulz[0][0] # we initialize a first histogram, representing the benchmark (Stulz)
bar2 = doubleModel(x_test).numpy().squeeze() + price_Stulz[0][0] # we initialize a second histogram, representing the algorithm

# we create the graph
fig_PnL = plt.figure(dpi= 125, facecolor='w')
ax = fig_PnL.add_subplot()
ax.set_xlabel("PnL", size = 15)
ax.set_ylabel("Count", size = 15)

# we overlap two histograms with different opacity, to make the visual interpretation easier
ax.hist((bar1), bins=100, label=["Stulz PnL"], color=["crimson"], alpha=0.8)
ax.hist((bar2), bins=100, label=["Deep Hedging PnL"], color=["pink"], alpha=0.4)
ax.legend(loc='upper left', prop={'size': 15}) # we pop a legend, differentiating the benchmark and algorithm PnLs

# the graph now looks cleaner: we delete its usual black framework
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.show()
#----------------------------------------------------------------------