# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:56:07 2019

@author: Jonat
"""
import numpy as np
from .processing import get_counts

def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

def r_squared(y, y_hat):
    SSres = np.sum((y_hat - y)**2)
    SStot = np.sum((y - np.mean(y))**2)
    return 1 - (SSres/SStot)

def gini(y, y_hat=None):
    counts   = get_counts(y)
    len_y    = len(y)
    impurity = 1 - sum((counts[key] / len_y)**2 for key in counts.keys())
    return impurity

def accuracy(y, y_hat):
    length = len(y)
    num_correct = np.sum(y == y_hat)
    return num_correct/length

def precision(y, y_hat):
    """ Precision for a classifier making a binary prediction"""
    pass