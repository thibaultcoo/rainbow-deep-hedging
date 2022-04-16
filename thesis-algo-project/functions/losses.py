import tensorflow.keras.backend as K # extension of Numpy for tensors (allows basic operations and computations)

def Entropy(wealth=None, loss_param=None):
    lamb = loss_param

    # Entropy (exponential) risk measure
    return (1/lamb)*K.log(K.mean(K.exp(-lamb*wealth)))

def MSE(wealth=None, nobs=None):

    # Mean squared error
    return (1/nobs)*K.sum(K.square(wealth))