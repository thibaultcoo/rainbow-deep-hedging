from tensorflow.keras.layers import Dense, Activation, ReLU, Concatenate
from tensorflow.keras.initializers import he_uniform # method used to initialize our weights and biases (see chap3)
import tensorflow as tf # backend of our global structure

class stat_layer(tf.keras.layers.Layer):
    def __init__(self, d=None, m=None, kernel_initializer=he_uniform(),
                 activation_dense="relu", activation_output="linear", day=None):

        super().__init__(name="delta_" + str(day)) # we inherit from the parent class "Layer" from keras

        self.d = d # number of hidden layers
        self.m = m # number of neurons per layer
        self.activation_dense = activation_dense # activation function applied to the hidden layers
        self.activation_output = activation_output # activation function applied to the output layer
        self.kernel_initializer = kernel_initializer
        self.intermediate_dense = [None for _ in range(d)] # initializing all hidden layers

        for i in range(d):
            self.intermediate_dense[i] = Dense(self.m, he_uniform(), bias_initializer=he_uniform())

        self.output_dense = Dense(1, kernel_initializer=he_uniform(), bias_initializer=he_uniform())

    def call(self, input):

        # the input goes in the hidden layer and is transformed
        for i in range(self.d):
            if i == 0:
                output = self.intermediate_dense[i](input)
            else:
                output = self.intermediate_dense[i](output)

            # the final output goes in the ouput layer and is finally transformed
            output = ReLU()(output)

        output = self.output_dense(output)
        output = Activation(self.activation_output)(output)

        return output # the final tensor is finally returned