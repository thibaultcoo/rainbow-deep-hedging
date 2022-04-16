from tensorflow.keras.layers import Input, Subtract, Lambda
from tensorflow.keras.layers import Add, Dot, Activation
from tensorflow.keras.models import Model # used to build the backbone of our algorithm
from tensorflow.keras.initializers import he_uniform # method used to initialize our weights and biases (see chap3)
import tensorflow.keras.backend as K # extension of numpy for tensors, allows basic and more complex operations
import numpy as np # used occasionnally when not dealing with tensors (but with scalars)
import tensorflow as tf # backend of our global structure

from rainbowAlgo.rainbow_layerGenerator import stat_layer

def rainbowModel(N = None, nb_neurons = None, nb_hidden = None,r = None, dt = None,
                 initial_wealth = 0.0, eps = 0.0, kernel_initializer = he_uniform(),
                 activation_dense = "relu", activation_output = "linear"):

    prc_1 = Input(shape=(1,), name="prc_1_0") # price of asset 1
    info_1 = Input(shape=(1,), name="information_set_1_0") # log-transformed and centered price of asset 1

    prc_2 = Input(shape=(1,), name="prc_2_0") # price of asset 2
    info_2 = Input(shape=(1,), name="information_set_2_0") # log-transformed and centered price of asset 2

    inputs = [prc_1, info_1, prc_2, info_2] # the entire set of inputs logically comprises both prices and info tensors
    info = [info_1, info_2] # info set solely comprises the transformed prices

    for j in range(N + 1): # for each time step, we build the corresponding delta-hedging strategy
        if j < N: # the final step is particular, since we end the hedging

            helper1 = info

            # we generate our two delta-hedging parameters through the output for each iteration of our model
            layer = stat_layer(m=nb_neurons, d=nb_hidden, kernel_initializer=he_uniform(), day=j,
                               activation_dense=activation_dense, activation_output=activation_output)

            strat_helper = layer(helper1) # we input the transformed prices into the model, which will then output our parameters

            if j == 0:
                delta_strat = strat_helper # for the first period, the strategy adopted is purely given by the new parameters
            else:
                # but after that, we will solely trade (buy or sell) the spread between the new strategy and the preceding one
                delta_strat = Subtract(name="diff_strategy_" + str(j))([strat_helper, strategy])

            # we separate the two output parameters produced by the algo to work with them separately
            delta_strat_1, delta_strat_2 = tf.split(delta_strat, num_or_size_splits=2, axis=1)

            # we first compute the cost of delta-hedging for both our underlying assets
            absChanges_1 = Lambda(lambda x: K.abs(x), name="absolutechanges_1_" + str(j))(delta_strat_1)
            c_1 = Dot(axes=1)([absChanges_1, prc_1])
            c_1 = Lambda(lambda x: eps * x, name="cost_1_" + str(j))(c_1)

            absChanges_2 = Lambda(lambda x: K.abs(x), name="absolutechanges_2_" + str(j))(delta_strat_2)
            c_2 = Dot(axes=1)([absChanges_2, prc_2])
            c_2 = Lambda(lambda x: eps * x, name="cost_2_" + str(j))(c_2)

            # we then subtract all trading costs from our final wealth
            if j == 0:
                w = Lambda(lambda x: initial_wealth - x, name="costDot_1_" + str(j))(c_1)
                w = Subtract(name="costDot_2_" + str(j))([w, c_2])
            else:
                w = Subtract(name="costDot_1_" + str(j))([w, c_1])
                w = Subtract(name="costDot_2_" + str(j))([w, c_2])

            # we then compute the final value of our portfolio, given we traded in both underlying assets to delta-hedge
            mult_1 = Dot(axes=1)([delta_strat_1, prc_1])
            w = Subtract(name="wealth_partly_" + str(j))([w, mult_1])

            mult_2 = Dot(axes=1)([delta_strat_2, prc_2])
            w = Subtract(name="wealth_final_" + str(j))([w, mult_2])

            # we consider the fair value of our portfolio
            fair_value_factor = np.exp(r * dt)
            w = Lambda(lambda x: x * fair_value_factor)(w)

            # we prepare the four input parameters that are to be used for the next iteration in our algorithm
            prc_1 = Input(shape=(1,), name="prc_1_" + str(j + 1))
            info_1 = Input(shape=(1,), name="information_set_1_" + str(j + 1))

            prc_2 = Input(shape=(1,), name="prc_2_" + str(j + 1))
            info_2 = Input(shape=(1,), name="information_set_2_" + str(j + 1))

            info = [info_1, info_2]

            strategy = strat_helper

            if j != N - 1:
                inputs += [prc_1, prc_2, info_1, info_2]
            else:
                inputs += [prc_1, prc_2]
        else:

            # if we are at the final period, we perform the same reasoning, only we will not prepare the inputs for the next iteration
            delta_strat_1, delta_strat_2 = tf.split(strategy, num_or_size_splits=2, axis=1)

            mult_1 = Dot(axes=1)([delta_strat_1, prc_1])
            w = Add()([w, mult_1])

            mult_2 = Dot(axes=1)([delta_strat_2, prc_2])
            w = Add()([w, mult_2])

            payoff = Input(shape=(1,), name="payoff")
            inputs += [payoff]

            # the final wealth is then defined as such (see chap2)
            w = Add(name="wealth_final_" + str(j))([w, payoff])

    # the model is called, with the four inputs given, as well as the wealth function as an output parameter (that is to be minimized through a loss function)
    return Model(inputs=inputs, outputs=w)