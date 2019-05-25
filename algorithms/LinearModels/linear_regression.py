# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:03:30 2019

@author: Jonat

Creates different versions of Linear Regression.
"""
import numpy as np
from utils import standardize

class LinearRegression():
    
    def __init__(self, n_iter=500, eta=.001, standardize=True, gd=True):
        self.n_iter      = n_iter
        self.eta         = eta
        self.standardize = standardize
        self.gd          = gd
        
    def predict(self, X):
        """
        Creates predicted value by multipltying the feature matrix by their weights, and adding the intercept term
        """
        return X @ (self.w[1:]) + self.w[0]
    
    def fit(self, X, y):
        """
        Determine statistical relationship between columns in X and target variable y
        """
        # standardize feature matrix if needed
        if self.standardize:
            X = standardize(X)
            
        # if gradient descent, then solve w/ closed form solution    
        if not self.gd:
            # add bias unit
            X      = np.c_[np.ones(len(X)), X]
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
            
        # otherwise, use gradient descent
        else:
            rgen = np.random.RandomState()
            # initialize weights, adding an extra for the intercept
            self.w      = rgen.normal(loc=0, scale=0.1, size=X.shape[1] + 1)
            self.cost_  = []
            
            for i in range(self.n_iter):
                output      = self.predict(X)           # create prediction
                errors      = y - output                # get errors
                gradient    = X.T @ errors * 1/len(X)   # get gradient w.r.t. each column, scale by # of samples
                self.w[1:] += gradient * self.eta       # update weights
                self.w[0]  += errors.sum() * self.eta * 1/len(X)
                cost        = np.sum(errors**2) / 2     # calculate cost
                self.cost_.append(cost)