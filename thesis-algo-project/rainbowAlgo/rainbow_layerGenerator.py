from tensorflow.keras.layers import Dense, Activation, ReLU, Concatenate
from tensorflow.keras.initializers import he_uniform
import tensorflow as tf

class stat_layer(tf.keras.layers.Layer):
    def __init__(self, d=None, m=None, kernel_initializer=he_uniform(),
                 activation_dense="relu", activation_output="linear", day=None):

        super().__init__(name="delta_" + str(day))

        self.d = d
        self.m = m
        self.activation_dense = activation_dense
        self.activation_output = activation_output
        self.kernel_initializer = kernel_initializer
        self.intermediate_dense = [None for _ in range(d)]

        for i in range(d):
            self.intermediate_dense[i] = Dense(self.m, input_dim=2, kernel_initializer=he_uniform(), #input_dim = 2
                                               bias_initializer=he_uniform())

        self.output_dense = Dense(2, kernel_initializer=he_uniform(),
                                  bias_initializer=he_uniform())

    def call(self, input):

        x1, x2 = input

        x = Concatenate(axis=1)([x1, x2])

        for i in range(self.d):
            if i == 0:
                output = self.intermediate_dense[i](x)
            else:
                output = self.intermediate_dense[i](output)

            output = ReLU()(output)

        output = self.output_dense(output)
        output = Activation(self.activation_output)(output)

        return output