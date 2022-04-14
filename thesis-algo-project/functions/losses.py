import tensorflow.keras.backend as K

def Entropy(wealth=None, loss_param=None):
    lamb = loss_param

    # Entropy (exponential) risk measure
    return (1/lamb)*K.log(K.mean(K.exp(-lamb*wealth)))

def CVaR(wealth = None, w = None, loss_param = None):
    alpha = loss_param

    # Expected shortfall risk measure
    return K.mean(w + (K.maximum(-wealth-w,0)/(1.0-alpha)))

def MSE(wealth=None):

    # Mean squared error
    return (1/120000)*K.sum(K.square(wealth))