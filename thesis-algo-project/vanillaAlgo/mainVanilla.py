#--------------------- importing the required libraries -----------------
from IPython.display import clear_output # cleaner outputs
import numpy as np # various calculations

from tensorflow.compat.v1.keras.optimizers import Adam # stochastic gradient descent algorithm used (see chap3)
import matplotlib.pyplot as plt # nice looking graphs

from options.EuropeanPut import EuropeanPut # class embedded with functions for its price, pnl and delta
from options.EuropeanCall import EuropeanCall # class embedded with functions for its price, pnl and delta
from vanillaAlgo.vanilla_generalAlgo import vanillaModel # class embedded with the general algorithm for the rainbow hedging
from functions.losses import Entropy # loss functions
from functions.splitVanilla import set_split_vanilla # will split our entire data sets into a training and testing set

from dynamics.vanillaDyn.singleGeomBrownian import GeometricBrownianMotion # used to generate regular Brownian motions paths
from dynamics.vanillaDyn.Heston import Heston # used to generate Heston paths
#------------------------------------------------------------------------
clear_output()
print("\nAll libraries are now imported.")
#------------------- initializing our constants -----------------------------
N = 30 # number of time steps
S0 = 100.0 # initial spot
sigma = 0.2 # volatility of the underlying's price
r = 0.0 # risk-free rate
T = 1 # maturity

Ktrain = 1*(10**3) # date points in the training set
Ktest_ratio = 0.2 # data points in the testing set as a share of the training set

strike = S0 # strike of our vanilla option
payoff_call = lambda x: -np.maximum(x - strike , 0.0) # call option payoff
payoff_put = lambda x: -np.maximum(strike - x, 0.0) # put option payoff

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

# calibrated Heston parameters (see chap4 for more explanation)
v0 = 0.0654
kappa = 0.6067
theta = 0.0707
xi = 0.2928
rho = -0.7571
#------------------------------------------------------------------------
process_BS = GeometricBrownianMotion(s0 = S0, sigma = sigma, r = r, matu=T, N=N, dt=dt)
#process_Heston = Heston(s0 = S0, v0 = v0, r = r, T = matu, N = N, kappa = kappa,
                        #theta = theta, xi = xi, rho = rho, dt = dt)

spot_BS = process_BS.gen_path(nobs) # dim -> (120000, 31)
#spot_Heston = process_Heston.gen_path(nobs)
#---------------------------------------------------------------------
clear_output()
print("\nAll price paths are now created.")
#----------------------- preparing our data set -------------------------
finalPayoff = payoff_call(spot_BS[:, -1]) # computes the payoffs at the final date of all paths
tradingSet =  np.stack((spot_BS), axis=1) # stacks all layers comprised of the spot paths
infoSet =  np.stack((np.log(spot_BS / S0)), axis=1) # stacks those same layers, log-transformed and centered on the strike

x_all = [] # initializing the matrix that will comprise the entire dataset
for i in range(N+1): # we will successively add the price and the log-transformed and centered price for the asset
  x_all += [tradingSet[i, :, None]]
  if i != N:
    x_all += [infoSet[i, :, None]]
x_all += [finalPayoff[:, None]] # we end up by adding the final payoffs layer

test_size = int(Ktrain*Ktest_ratio) # we denote as such the size of the testing set

# we split the data into our two sets
[x_train, x_test] = set_split_vanilla(x_all, test_size=test_size)
[S_train, S_test] = set_split_vanilla([spot_BS], test_size=test_size)
[payoff_train, payoff_test] = set_split_vanilla([x_all[-1]], test_size=test_size)
#----------------------------------------------------------------------
print("\nAll data sets are now prepared.")
#--------------------- running the algorithm --------------------------
optimizer = Adam(learning_rate=lr) # initializing the gradient descent algorithm with the chosen learning rate

simpleModel = vanillaModel(N=N, nb_neurons=nb_neurons, nb_hidden=nb_hidden+2, r=r, dt=dt, eps=eps,
                           kernel_initializer = kernel_initializer, activation_dense = activation_dense,
                           activation_output = activation_output)

loss = Entropy(simpleModel.output, loss_param=1.0)
simpleModel.add_loss(loss)

simpleModel.compile(optimizer=optimizer) # compiling the model

# we run the neural networks under the afordefined stochastic paths
simpleModel.fit(x=x_train, batch_size=batch_size, epochs=epochs, validation_data=x_test, verbose=1)
#------------------------------------------------------------------------
clear_output()
print("\nThe algorithm has now finished running.")
print("\nThe Deep Hedging strategy, as well as the predicted price of the vanilla option, will be shown soon.")
#---------------------- benchmark comparison ---------------------------
option = EuropeanPut()

# we compute the price of our benchmark (Black-Scholes)
price_BS = option.put_price(S = S_test[0], sigma = sigma, r = r, K = strike, N = N, dt = dt)

# we compute the delta-hedging parameter of our benchmark (Black-Scholes)
delta_BS = option.put_delta(S = S_test[0], sigma = sigma, r = r, K = strike, N = N, dt = dt)

# we compute the overall PnL of our benchmark (Black-Scholes)
PnL_BS =  option.put_pnl(S = S_test[0], payoff= payoff_put(S_test[0][:, -1]), delta=delta_BS,
                         dt= dt, r = r, eps=eps)

mc_price = -payoff_test[0].mean() * np.exp(-r * (N * dt))
algo_price = simpleModel.evaluate(x_test, batch_size=test_size, verbose=0)
#------------------------------------------------------------------------
print("The Black-Scholes model price is %2.3f." % price_BS[0][0])
print("The Monte-Carlo price is %2.3f." % mc_price)
print("The Deep Hedging price is %2.3f." % algo_price)
#------------------- building the comparison graph --------------------------
bar1 = PnL_BS + price_BS[0][0] # we initialize a first histogram, representing the benchmark (Black-Scholes)
bar2 = simpleModel(x_test).numpy().squeeze() + price_BS[0][0] # we initialize a second histogram, representing the algorithm

# we create the graph
fig_PnL = plt.figure(dpi= 125, facecolor='w')
ax = fig_PnL.add_subplot()
ax.set_xlabel("PnL", size = 15)
ax.set_ylabel("Count", size = 15)

# we overlap two histograms with different opacity, to make the visual interpretation easier
ax.hist((bar1), bins=100, label=["Black-Scholes PnL"], color=["midnightblue"], alpha=0.8)
ax.hist((bar2), bins=100, label=["Deep Hedging PnL"], color=["c"], alpha=0.4)
ax.legend(loc='upper left', prop={'size': 15}) # we pop a legend, differentiating the benchmark and algorithm PnLs

# the graph now looks cleaner: we delete its usual black framework
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.show()
#------------------------------------------------------------------------