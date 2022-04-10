#--------------------- importing the required libraries -----------------
from IPython.display import clear_output
import numpy as np

from tensorflow.compat.v1.keras.optimizers import Adam
import matplotlib.pyplot as plt

from options.EuropeanPut import EuropeanPut
from vanillaAlgo.vanilla_generalAlgo import vanillaModel
from functions.losses import Entropy
from functions.splitVanilla import set_split_vanilla

from dynamics.vanillaDyn.singleGeomBrownian import GeometricBrownianMotion
from dynamics.vanillaDyn.Heston import Heston
#------------------------------------------------------------------------
clear_output()
print("\nFinish installing and importing all necessary libraries!")
#------------------- initializing our constants -----------------------------
N = 30
S0 = 100.0
sigma = 0.2
r = 0.0
T = 1

Ktrain = 1*(10**5)
Ktest_ratio = 0.2

strike = S0
payoff_call = lambda x: -np.maximum(x - strike , 0.0) # call option payoff
payoff_put = lambda x: -np.maximum(strike - x, 0.0) # put option payoff

eps = 0.0
nb_neurons = 25
nb_hidden = 2

lr = 0.001 # learning rate
batch_size = 200 # batch size
epochs = 5 # epochs
kernel_initializer = "he_uniform"

activation_dense = "relu"
activation_output = "sigmoid"

seed = 0
nobs = int(Ktrain*(1+Ktest_ratio)) 

dt = 1/365

# calibrated Heston parameters
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

clear_output()
#----------------------- preparing our data set -------------------------
finalPayoff = payoff_call(spot_BS[:, -1]) # dim -> (120000,)
tradingSet =  np.stack((spot_BS), axis=1) # dim -> (31, 120000)
infoSet =  np.stack((np.log(spot_BS / S0)), axis=1) # dim -> (31, 120000)

x_all = [] # dim -> (62, 120000, 1)
for i in range(N+1):
  x_all += [tradingSet[i, :, None]]
  if i != N:
    x_all += [infoSet[i, :, None]]
x_all += [finalPayoff[:, None]]

test_size = int(Ktrain*Ktest_ratio) # 20000
[x_train, x_test] = set_split_vanilla(x_all, test_size=test_size) # dim respectively -> (62, 100000, 1) (62, 20000, 1)
[S_train, S_test] = set_split_vanilla([spot_BS], test_size=test_size) # dim respectively -> (1, 100000, 31) (1, 20000, 31)
[payoff_train, payoff_test] = set_split_vanilla([x_all[-1]], test_size=test_size) # dim respectively -> (1, 100000, 1) (1, 20000, 1)
#----------------------------------------------------------------------
print("Finish preparing data!")
#--------------------- running the algorithm --------------------------
optimizer = Adam(learning_rate=lr)

simpleModel = vanillaModel(N=N, nb_neurons=nb_neurons, nb_hidden=nb_hidden+2, r=r, dt=dt, eps=eps,
                           kernel_initializer = kernel_initializer, activation_dense = activation_dense,
                           activation_output = activation_output)

loss = Entropy(simpleModel.output, loss_param=1.0)
simpleModel.add_loss(loss)

simpleModel.compile(optimizer=optimizer)

simpleModel.fit(x=x_train, batch_size=batch_size, epochs=epochs, validation_data=x_test, verbose=1)
#------------------------------------------------------------------------
clear_output()

print("Finished running deep hedging algorithm! (Simple Network)")
#---------------------- benchmark comparison ---------------------------
option = EuropeanPut()

price_BS = option.put_price(S = S_test[0], sigma = sigma, r = r, K = strike, N = N, dt = dt)

delta_BS = option.put_delta(S = S_test[0], sigma = sigma, r = r, K = strike, N = N, dt = dt)

PnL_BS =  option.put_pnl(S = S_test[0], payoff= payoff_put(S_test[0][:, -1]), delta=delta_BS,
                         dt= dt, r = r, eps=eps)

risk_neutral_price = -payoff_test[0].mean() * np.exp(-r * (N * dt))
nn_simple_price = simpleModel.evaluate(x_test, batch_size=test_size, verbose=0)
#------------------------------------------------------------------------
print("The Black-Scholes model price is %2.3f." % price_BS[0][0])
print("The Risk Neutral price is %2.3f." % risk_neutral_price)
print("The Deep Hedging price is %2.3f." % nn_simple_price)
#------------------- building the comparison graph --------------------------
bar1 = PnL_BS + price_BS[0][0]
bar2 = simpleModel(x_test).numpy().squeeze() + price_BS[0][0]

fig_PnL = plt.figure(dpi= 125, facecolor='w')
ax = fig_PnL.add_subplot()
ax.set_xlabel("PnL", size = 15)
ax.set_ylabel("Count", size = 15)
ax.hist((bar1), bins=100, label=["Black-Scholes PnL"], color=["midnightblue"], alpha=0.8)
ax.hist((bar2), bins=100, label=["Deep Hedging PnL"], color=["c"], alpha=0.4)
ax.legend(loc='upper left', prop={'size': 15})

right_side = ax.spines["right"]
right_side.set_visible(False)

top_side = ax.spines["top"]
top_side.set_visible(False)

plt.show()
#------------------------------------------------------------------------