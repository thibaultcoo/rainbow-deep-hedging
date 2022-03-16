#!/usr/bin/env python
# coding: utf-8

import tensorflow.keras.backend as K

def Entropy(w = None, loss_param = None):
    return (1/loss_param)*K.log(K.mean(K.exp(-loss_param * w)))

def CVaR(wealth = None, w = None, loss_param = None):
    return K.mean(w + (K.maximum(-wealth-w, 0)/(1.0-loss_param)))

def MSE(w = None):
    return K.mean(K.square(w))