from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, LeakyReLU
from tensorflow.keras.layers import Lambda, Add, Dot, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal, Zeros, he_uniform, TruncatedNormal
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

# BN -> batch normalization
# d -> number of hidden layers - including input layer
# m -> number of neurons in each hidden layers

# class function that returns a single layer for the algorithm, is to be called iteratively
class vanillaStratLayer(tf.keras.layers.Layer):
    
    def __init__(self, d = None, m = None, use_batch_norm = None, kernel_initializer = he_uniform(), activation_dense = "relu", activation_output = "linear", day = None):
        
        super().__init__(name = "delta_" + str(day))
        self.d = d
        self.m = m
        self.use_batch_norm = use_batch_norm
        self.activation_dense = activation_dense # activation function of the hidden layers
        self.activation_output = activation_output # activation function of the output layer
        self.kernel_initializer = kernel_initializer # weights/biases initializer
        
        self.intermediate_dense = [None for _ in range(d)]
        self.intermediate_BN = [None for _ in range(d)]

        # we fill the hidden layers as Dense layers
        for i in range(d):
            self.intermediate_dense[i] = Dense(self.m, kernel_initializer = self.kernel_initializer,
                                               bias_initializer = he_uniform(), use_bias = (not self.use_batch_norm))
            if self.use_batch_norm:
                self.intermediate_BN[i] = BatchNormalization(momentum = 0.99, trainable = True)

        # the output layer is also a dense layer but with a single neuron
        self.output_dense = Dense(1, kernel_initializer = self.kernel_initializer, bias_initializer = he_uniform(), use_bias = True)

    # dont really know what it does
    def call(self, input):
        
        for i in range(self.d):
            if i == 0:
                output = self.intermediate_dense[i](input)
            else:
                output = self.intermediate_dense[i](output)
                
            if self.use_batch_norm:
                output = self.intermediate_BN[i](output, training = True)
                
            if self.activation_dense == "leakyReLU":
                output = LeakyReLU()(output)
            else:
                output = Activation(self.activation_dense)(output)
            
        output = self.output_dense(output)
        
        if self.activation_output == "leakyReLU":
            output = LeakyReLU()(output)
        else:
            output = Activation(self.activation_output)(output)
        
        return output

