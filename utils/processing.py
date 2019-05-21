"""
Created on Sun Mar 31 10:53:17 2019

class of helper functions to make core code tidier

@author: Jonat
"""
import numpy as np

def standardize(X):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X

def polynomialize(X, degree=2):
    
    if degree < 2:
        raise ValueError("degree argument must be 2 or larger")
        
    for col in range(X.shape[1]):
        for exp in range(2, degree+1):
            col_to_add = X[:, col]**exp
            col_to_add = col_to_add[:, np.newaxis]
            X = np.hstack((X, col_to_add))
    return X

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
    
def sigmoid(X):
    return 1 / (1 + np.exp(-X))