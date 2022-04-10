from tensorflow.keras.layers import Input, Subtract, Lambda
from tensorflow.keras.layers import Add, Dot, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_uniform
import tensorflow.keras.backend as K
import numpy as np

from vanillaAlgo.vanilla_layerGenerator import stat_layer

def vanillaModel(N = None, nb_neurons = None, nb_hidden = None,r = None, dt = None,
                 initial_wealth = 0.0, eps = 0.0, kernel_initializer = he_uniform(),
                 activation_dense = "relu", activation_output = "linear"):

    prc = Input(shape=(1,), name = "prc_0")
    info = Input(shape=(1,), name = "information_set_0")

    inputs = [prc, info]
    
    for j in range(N+1):            
        if j < N:

            helper1 = info

            layer = stat_layer(m = nb_neurons, d = nb_hidden, kernel_initializer = he_uniform(), day = j,
                                activation_dense = activation_dense, activation_output = activation_output)

            strat_helper = layer(helper1)

            if j == 0:
                delta_strat = strat_helper
            else:
                delta_strat = Subtract(name = "diff_strategy_" + str(j))([strat_helper, strategy])

            absChanges = Lambda(lambda x : K.abs(x), name = "absolutechanges_" + str(j))(delta_strat)

            c = Dot(axes=1)([absChanges,prc])
            c = Lambda(lambda x : eps * x, name ="cost_" + str(j))(c)

            if j == 0:
                w = Lambda(lambda x : initial_wealth - x, name = "costDot_" + str(j))(c)
            else:
                w = Subtract(name = "costDot_" + str(j))([w, c])

            mult = Dot(axes=1)([delta_strat, prc])
            w = Subtract(name = "wealth_" + str(j))([w, mult])

            FV_factor = np.exp(r * dt)
            w = Lambda(lambda x: x*FV_factor)(w)
            
            prc = Input(shape=(1,), name = "prc_" + str(j+1))
            info = Input(shape=(1,), name = "information_set_" + str(j+1))
            
            strategy = strat_helper
            
            if j != N - 1:
                inputs += [prc, info]
            else:
                inputs += [prc]
        else:

            mult = Dot(axes=1)([strategy, prc])
            w = Add()([w, mult])

            payoff = Input(shape=(1,), name = "payoff")
            inputs += [payoff]
            
            w = Add(name = "wealth_" + str(j))([w,payoff])

    return Model(inputs=inputs, outputs=w)