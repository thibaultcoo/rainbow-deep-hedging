from tensorflow.keras.layers import Input, Subtract, Lambda
from tensorflow.keras.layers import Add, Dot, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_uniform
import tensorflow.keras.backend as K
import numpy as np

from rainbowAlgo.rainbow_layerGenerator import strat_layer

# we define the rainbowModel
# general algo