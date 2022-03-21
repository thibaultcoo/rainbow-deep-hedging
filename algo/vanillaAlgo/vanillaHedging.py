from tensorflow.keras.layers import Input, Concatenate, Subtract
from tensorflow.keras.layers import Lambda, Add, Dot
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

from algo.vanillaAlgo.stratlayer import vanillaStratLayer

# pretty much understand everything that is going on here
# crucial to detect how dealing with two underlying assets will impact this function
def vanillaHedging(N = None, d = None, m = None, r = None, dt = None, init_w = None, eps = None, T_cost = False, strat_type = None, use_batch_norm = None, \
            kernel_initializer = "he_uniform", activation_dense = "relu", activation_output = "linear", share_strat_across_time = False):
    
    price = Input(shape=(1,), name = "price_0")
    info_set = Input(shape=(1,), name = "info_set_0")
    
    inputs = [price, info_set]
    
    for j in range(N+1):
        if j < N:
            if strat_type == "simple":
                helper1 = info_set
            
            if not share_strat_across_time:
                strat_layer = vanillaStratLayer(d = d, m = m, use_batch_norm = use_batch_norm, kernel_initializer = kernel_initializer, \
                                         activation_dense = activation_dense, activation_output = activation_output, day = j)
                
            else:
                if j == 0:
                    strat_layer = vanillaStratLayer(d = d, m = m, use_batch_norm = use_batch_norm, kernel_initializer = kernel_initializer, \
                                             activation_dense = activation_dense, activation_output = activation_output, day = j)

            stratHelper = strat_layer(helper1)
            
            if j == 0:
                delta_strat = stratHelper
            else:
                delta_strat = Subtract(name = "diff_strategy_" + str(j))([stratHelper, strat])
            
            absChanges = Lambda(lambda x : K.abs(x), name = "absChanges_" + str(j))(delta_strat)
            costs = Dot(axes=1)([absChanges, price])
            costs = Lambda(lambda x : eps*x, name = "costs_" + str(j))(costs)
            
            if j == 0:
                w = Lambda(lambda x : init_w - x, name = "costsDot_" + str(j))(costs)
            else:
                w = Subtract(name = "costsDot_" + str(j))([w, costs])
                
            # wealth for the next period -> previous wealth - cost of today's delta hedge strategy
            mult = Dot(axes=1)([delta_strat, price])
            w = Subtract(name = "w_" + str(j))([w, mult])
            w = Lambda(lambda x : x * np.exp(r * dt))(w)
            
            price = Input(shape=(1,), name = "price_" + str(j+1))
            info_set = Input(shape=(1,), name = "info_set" + str(j+1))
            
            strat = stratHelper
            
            # we materialize the gain in info along the time steps
            if j != N-1:
                inputs = inputs + [price, info_set]
            else:
                inputs = inputs + [price]
            
        else:
            if T_cost:
                absChanges = Lambda(lambda x : K.abs(x), name = "absChanges_" + str(j))(strat)
                costs = Dot(axes=1)([absChanges, price])
                costs = Lambda(lambda x : eps * x, name = "costs_" + str(j))(costs)
                w = Subtract(name = "costsDot_" + str(j))([w, costs])

            # final period wealth
            mult = Dot(axes=1)([strat, price])
            w = Add()([w, mult])
            
            # adding the terminal payoff of the derivative
            payoff = Input(shape=(1,), name = "payoff")
            inputs = inputs + [payoff]
            w = Add(name = "w_" + str(j))([w, payoff])

    return Model(inputs=inputs, outputs=w)

#partie RNN
#généralisation des ANN aux séries temporelles
#un peu plus intéressante à détailler (faut vraiment qu'on soit carré là-dessus)
#faut garder les RNN comme un modèle joker (le garder comme une ouverture)
#pré-requis: variables qui sont les signaux de trading
#l'idée serait d'éventuellement: en quoi consistent ces signaux
#peut on utiliser des signaux/forecasts de volatilité et de corrélation pour hedger des rainbows