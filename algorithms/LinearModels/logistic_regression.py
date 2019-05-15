# -*- coding: utf-8 -*-
"""
Logistic Regression, with an optional L2 norm, written in Python/Numpy.
"""

import numpy as np
from utils import standardize

class LogisticRegression():
    
    def __init__(self, n_iter=500, eta=.001, standardize=True):
        self.n_iter      = n_iter
        self.eta         = eta
        self.standardize = standardize
        
    def predict(self, X):
        """
        Creates predicted value by multipltying the feature matrix by their weights, and adding the intercept term
        """
        return X @ (self.w[1:]) + self.w[0]
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def fit(self, X, y):
        """
        Determine statistical relationship between columns in X and target variable y
        """
        # standardize feature matrix if needed
        if self.standardize:
            X = standardize(X)
            
        rgen = np.random.RandomState()
        # initialize weights, adding an extra for the intercept
        self.w      = rgen.normal(loc=0, scale=0.1, size=X.shape[1] + 1)
        self.cost_  = []
            
        for i in range(self.n_iter):
            output      = self.sigmoid(self.predict(X))      # create prediction
            errors      = y - output                         # get errors
            gradient    = X.T @ errors * 1/len(X)            # get gradient w.r.t. each column
            self.w[1:] += gradient * self.eta                # update weights
            self.w[0]  += errors.sum() * self.eta
            cost        = np.sum(errors**2) / 2              # calculate cost
            self.cost_.append(cost)