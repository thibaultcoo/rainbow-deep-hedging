#!/usr/bin/env python
# coding: utf-8

from sklearn import model_selection

def train_test_split(data = None, test_size = None):
    
    x_train = []
    x_test = []
    
    for x in data:
        temp_x_train, temp_x_test = \
            model_selection.train_test_split(x, test_size = test_size, shuffle = False)
        x_train = x_train + [temp_x_train]
        x_test = x_test + [temp_x_test]

    return x_train, x_test