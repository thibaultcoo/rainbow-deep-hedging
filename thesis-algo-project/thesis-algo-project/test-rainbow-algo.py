from tensorflow.keras.layers import Input, Subtract, Lambda
from tensorflow.keras.layers import Add, Dot, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_uniform
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

from rainbowAlgo.rainbow_layerGenerator import stat_layer

# pretty great version, running but giving shit results (but running!)

def rainbowModel(N = None, nb_neurons = None, nb_hidden = None,r = None, dt = None,
                 initial_wealth = 0.0, eps = 0.0, kernel_initializer = he_uniform(),
                 activation_dense = "relu", activation_output = "linear"):

    prc = Input(shape=(1,), name="prc_0")
    info = Input(shape=(1,), name="information_set_0")

    inputs = [prc, info]

    for j in range(N + 1):
        if j < N:

            helper1 = info

            layer = stat_layer(m=nb_neurons, d=nb_hidden, kernel_initializer=he_uniform(), day=j,
                               activation_dense=activation_dense, activation_output=activation_output)

            strat_helper = layer(helper1)

            if j == 0:
                delta_strat = strat_helper
            else:
                delta_strat = Subtract(name="diff_strategy_" + str(j))([strat_helper, strategy])

            # the output is two-dimensional -> two hedging parameters -> we split it to isolate both
            delta_strat_1, delta_strat_2 = tf.split(delta_strat, num_or_size_splits=2, axis=1)

            absChanges_1 = Lambda(lambda x: K.abs(x), name="absolutechanges_1_" + str(j))(delta_strat_1)

            c_1 = Dot(axes=1)([absChanges_1, prc])
            c_1 = Lambda(lambda x: eps * x, name="cost_1_" + str(j))(c_1)

            absChanges_2 = Lambda(lambda x: K.abs(x), name="absolutechanges_2_" + str(j))(delta_strat_2)

            c_2 = Dot(axes=1)([absChanges_2, prc])
            c_2 = Lambda(lambda x: eps * x, name="cost_2_" + str(j))(c_2)

            if j == 0:
                w = Lambda(lambda x: initial_wealth - x, name="costDot_1_" + str(j))(c_1)
                w = Lambda(lambda x: initial_wealth - x, name="costDot_2_" + str(j))(c_2)
            else:
                w = Subtract(name="costDot_1_" + str(j))([w, c_1])
                w = Subtract(name="costDot_2_" + str(j))([w, c_2])

            mult_1 = Dot(axes=1)([delta_strat_1, prc])
            w = Subtract(name="wealth_1_" + str(j))([w, mult_1])

            mult_2 = Dot(axes=1)([delta_strat_2, prc])
            w = Subtract(name="wealth_2_" + str(j))([w, mult_2])

            FV_factor = np.exp(r * dt)
            w = Lambda(lambda x: x * FV_factor)(w)

            prc = Input(shape=(1,), name="prc_" + str(j + 1))
            info = Input(shape=(1,), name="information_set_" + str(j + 1))

            strategy = strat_helper

            if j != N - 1:
                inputs += [prc, info]
            else:
                inputs += [prc]
        else:

            delta_strat_1, delta_strat_2 = tf.split(strategy, num_or_size_splits=2, axis=1)

            mult_1 = Dot(axes=1)([delta_strat_1, prc])
            w = Add()([w, mult_1])

            mult_2 = Dot(axes=1)([delta_strat_2, prc])
            w = Add()([w, mult_2])

            payoff = Input(shape=(1,), name="payoff")
            inputs += [payoff]

            w = Add(name="wealth_" + str(j))([w, payoff])

    return Model(inputs=inputs, outputs=w)