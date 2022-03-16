from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, ReLU
from tensorflow.keras.layers import Lambda, Add, Dot, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal, Zeros, he_uniform, TruncatedNormal
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

# still dont know what that thing does
def vanillaDeltaModel(model = None, days_from_today = None, share_strategy_across_time = False, strat_type = "simple"):
    
    if strat_type == "simple":
        inputs = model.get_layer("delta_" + str(days_from_today)).input
        intermediate_inputs = inputs
    elif strat_type == "recurrent":
        inputs = [Input(1,), Input(1,)]
        intermediate_inputs = Concatenate()(inputs)
        
    if not share_strategy_across_time:
        outputs = model.get_layer("delta_" + str(days_from_today))(intermediate_inputs)
    else:
        outputs = model.get_layer("delta_0")(intermediate_inputs)
        
    return Model(inputs, outputs)

