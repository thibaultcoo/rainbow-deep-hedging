#--------------------- importing the required libraries -------------
from IPython.display import clear_output
import numpy as np

from tensorflow.compat.v1.keras.optimizers import Adam
import matplotlib.pyplot as plt

from options.EuropeanWorstofTwoCall import EuropeanWorstofTwoCall
from rainbowAlgo.rainbow_generalAlgo import rainbowModel
from functions.losses import Entropy, MSE
from functions.splitRainbow import set_split_rainbow

from dynamics.rainbowDyn.multiGeomBrownian import multiGeometric
from dynamics.rainbowDyn.kouJumpDiffusion import multiJumpDiffusion
#---------------------------------------------------------------------
clear_output()
print("\nFinish installing and importing all necessary libraries!")
#------------------- initializing our constants ----------------------
N = 30
T = 1
S0 = (100.0, 100.0)
m = (0.1, 0.1)
cov = np.zeros((2,2))
cov[0,0] = 0.05
cov[1,1] = 0.05
cov[0,1] = 0.03
cov[1,0] = 0.03
r = 0.0

Ktrain = 1*(10**5)
Ktest_ratio = 0.2

strike = np.mean(S0)
payoff_rainbow = lambda x, y: -np.maximum(0, np.minimum(x-strike, y-strike))

eps = 0.0
nb_neurons = 25
nb_hidden = 2

lr = 0.005
batch_size = 500
epochs = 10
kernel_initializer = "he_uniform"

activation_dense = "relu"
activation_output = "sigmoid"

seed = 0
nobs = int(Ktrain*(1+Ktest_ratio))

dt = 1/365

prob1 = 0.20
prob2 = 0.20

#---------------------------------------------------------------------
#process_Geom = multiGeometric(s0=S0, T=T, N=N, cov=cov, dt=dt)
process_Kou = multiJumpDiffusion(m=m, s0=S0, T=T, N=N, cov=cov, prob1=prob1, prob2=prob2, dt=dt)

#spot_Geom = process_Geom.gen_path(nobs)
spot_Geom = process_Kou.gen_path(nobs)

print(spot_Geom)

clear_output()
#---------------------------------------------------------------------
finalPayoff = payoff_rainbow(spot_Geom[:, -1, 0], spot_Geom[:, -1, 1])
tradingSet = np.stack((spot_Geom), axis=1)
infoSet = np.stack((np.log(spot_Geom / strike)), axis=1)

x_all = []
for i in range(N+1):
    for j in range(2):
        x_all += [tradingSet[i, :, j, None]]
        if i != N:
            x_all += [infoSet[i, :, j, None]]
x_all += [finalPayoff[:, None]]

print(np.shape(x_all))

test_size = int(Ktrain*Ktest_ratio)
[x_train, x_test] = set_split_rainbow(x_all, test_size=test_size)
[s_train, S_test] = set_split_rainbow([spot_Geom], test_size=test_size)
[payoff_train, payoff_test] = set_split_rainbow([x_all[-1]], test_size=test_size)

#----------------------------------------------------------------------
print("Finish preparing data!")
#--------------------- running the algorithm --------------------------
optimizer = Adam(learning_rate=lr)

doubleModel = rainbowModel(N=N, nb_neurons=nb_neurons, nb_hidden=nb_hidden+2, r=r, dt=dt, eps=eps,
                           kernel_initializer = kernel_initializer, activation_dense = activation_dense,
                           activation_output = activation_output)

#loss = Entropy(doubleModel.output, loss_param=1.0)
loss = MSE(doubleModel.output)
doubleModel.add_loss(loss)

doubleModel.compile(optimizer=optimizer)

doubleModel.fit(x=x_train, batch_size=batch_size, epochs=epochs, validation_data=x_test, verbose=1)
#----------------------------------------------------------------------
clear_output()

print("Finished running deep hedging algorithm! (Simple Network)")
#---------------------- benchmark comparison --------------------------
option = EuropeanWorstofTwoCall()

price_Stulz = option.get_Stulz_price(S=S_test[0], cov=cov, r=r, K=strike, matu=T,
                                     N=N, nbPaths=1, dt=dt)

delta_Stulz = option.get_Stulz_delta(S=S_test[0], cov=cov, r=r, K=strike, matu=T,
                                     N=N, nbPaths=1, dt=dt)

PnL_Stulz = option.get_Stulz_PnL(S=S_test[0],
                                 payoff=payoff_rainbow(S_test[0][:, -1, 0], S_test[0][:, -1, 1]),
                                 delta=delta_Stulz, matu=T, r=r, eps=eps, N=N, dt=dt)

risk_neutral_price = -payoff_test[0].mean() * np.exp(-r * (N * dt))
nn_double_price = doubleModel.evaluate(x_test, batch_size=test_size, verbose=0)
#----------------------------------------------------------------------
print("The Stulz model price is %2.3f." % price_Stulz[0][0])
print("The Monte-Carlo price is %2.3f." % risk_neutral_price)
print("The Deep Hedging price is %2.3f." % nn_double_price)
#------------------- building the comparison graph --------------------
bar1 = PnL_Stulz + price_Stulz[0][0]
bar2 = doubleModel(x_test).numpy().squeeze() + price_Stulz[0][0]

fig_PnL = plt.figure(dpi= 125, facecolor='w')
ax = fig_PnL.add_subplot()
ax.set_xlabel("PnL", size = 15)
ax.set_ylabel("Count", size = 15)
ax.hist((bar1), bins=100, label=["Stulz PnL"], color=["crimson"], alpha=0.8)
ax.hist((bar2), bins=100, label=["Deep Hedging PnL"], color=["pink"], alpha=0.4)
ax.legend(loc='upper left', prop={'size': 15})

right_side = ax.spines["right"]
right_side.set_visible(False)

top_side = ax.spines["top"]
top_side.set_visible(False)

plt.show()
#----------------------------------------------------------------------