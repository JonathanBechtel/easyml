# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:53:17 2019

class of helper functions to make core code tidier

@author: Jonat
"""
import numpy as np

def standardize(X):
    return (X - X.mean()) / X.std()

def polynomialize(X, degree=2):
    
    if degree < 2:
        raise ValueError("degree argument must be 2 or larger")
        
    if type(degree) is not int:
        raise TypeError("the degree argument must be an integer")
        
    columns = X.columns
    for column in columns:
        for exponent in range(2, degree+2):
            X[column+'^'+exponent] = X[column]**exponent

def shuffle_matrix(X, sample_size=1):
        num_rows = int(sample_size * X.shape[0])
        return np.random.permutation(X)[:num_rows]
    
def get_counts(col):
    unique, counts = np.unique(col, return_counts=True)
    counter        = dict(zip(unique, counts))
    return counter

def majority_vote(col):
    counts = np.bincount(list(col))
    winner = np.argmax(counts)
    return winner
    