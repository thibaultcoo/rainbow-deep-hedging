from tensorflow.keras.layers import Input, Subtract, Lambda
from tensorflow.keras.layers import Add, Dot, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_uniform
import tensorflow.keras.backend as K
import numpy as np

from rainbowAlgo.rainbow_layerGenerator import stat_layer

def rainbowModel(N = None, nb_neurons = None, nb_hidden = None,r = None, dt = None,
                 initial_wealth = 0.0, eps = 0.0, kernel_initializer = he_uniform(),
                 activation_dense = "relu", activation_output = "linear"):

    prc_1 = Input(shape=(2,), name = "prc_1_0")
    info_1 = Input(shape=(2,), name = "information_set_1_0")
    #prc_2 = Input(shape=(1,), name = "prc_2_0")
    #info_2 = Input(shape=(1,), name = "information_set_2_0")

    inputs = [prc_1, info_1]

    for j in range(N+1):
        if j < N:

            helper1 = [info_1] # needs to gather both info sets

            layer = stat_layer(m = nb_neurons, d = nb_hidden, kernel_initializer = he_uniform(), day = j,
                                activation_dense = activation_dense, activation_output = activation_output)

            strat_helper = layer(helper1)

            # some problems later that we will need to check
            # we will write what is following purposely wrong, but just to get the idea

            if j == 0:
                delta_strat_1 = strat_helper[1]
                delta_strat_2 = strat_helper[2]
            else:
                delta_strat_1 = Subtract(name = "diff_strategy_" + str(j))([strat_helper[1], strategy[1]])
                delta_strat_2 = Subtract(name = "diff_strategy_" + str(j))([strat_helper[2], strategy[2]])

            absChanges_1 = Lambda(lambda x : K.abs(x), name = "absolutechanges_" + str(j))(delta_strat_1)
            absChanges_2 = Lambda(lambda x : K.abs(x), name = "absolutechanges_" + str(j))(delta_strat_2)

            c1 = Dot(axes=1)([absChanges_1, prc_1])
            c2 = Dot(axes=1)([absChanges_2, prc_2])

            c1 = Lambda(lambda x : eps * x, name="cost_1_" + str(j))(c1)
            c2 = Lambda(lambda x: eps * x, name="cost_2_" + str(j))(c2)

            if j == 0:
                w = Lambda(lambda x, y : initial_wealth - x - y, name = "costDot_1_" + str(j))([c1, c2])
            else:
                w = Subtract(name = "costDot_" + str(j))([w, c1])
                w = Subtract(name = "costDot_" + str(j))([w, c2])

            mult = Dot(axes=1)([delta_strat_1, prc_1])
            w = Subtract(name="wealth_" + str(j))([w, mult])

            mult = Dot(axes=1)([delta_strat_2, prc_2])
            w = Subtract(name = "wealth_" + str(j))([w, mult])

            FV_factor = np.exp(r * dt)
            w = Lambda(lambda x: x*FV_factor)(w)

            prc_1 = Input(shape=(1,), name = "prc_1_" + str(j+1))
            info_1 = Input(shape=(1,), name = "information_set_1_" + str(j+1))

            prc_2 = Input(shape=(1,), name = "prc_2_" + str(j+1))
            info_2 = Input(shape=(1,), name = "information_set_2_" + str(j+1))

            strategy = strat_helper

            if j != N - 1:
                inputs += [prc_1, info_1, prc_2, info_2]
            else:
                inputs += [prc_1, prc_2]
        else:

            mult = Dot(axes=1)([strategy[1], prc_1])
            w = Add()([w, mult])

            mult = Dot(axes=1)([strategy[2], prc_2])
            w = Add()([w, mult])

            payoff = Input(shape=(1,), name="payoff")
            inputs += [payoff]

            w = Add(name="wealth_" + str(j))([w, payoff])

    return Model(inputs=inputs, outputs=w)