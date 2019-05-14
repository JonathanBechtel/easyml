# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:03:30 2019

@author: Jonat

Creates different versions of Linear Regression.
"""
import numpy as np
from utils import standardize

class LinearRegression():
    
    def __init__(self, n_iter=1000, eta=.001, standardize=True, gd=True):
        self.n_iter         = n_iter
        self.eta            = .001
        self.standardize    = standardize
        self.gd             = gd
        
     # create estimate of y   
    def output(self, X):
        return X.dot(self.w[1:]) + self.w[0]
    
    # determine statistical relationship between X and y
    def fit(self, X, y):
        
        # center w/ mean 0, var 1
        if self.standardize:
            X = standardize(X)
            
        # closed form of OLS    
        if not self.gd:
            # add bias unit
            X      = np.c_[np.ones(len(X)), X]
            # solve matrix equation
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        
        # gradient descent
        else:
           rgen        = np.random.RandomState()
           self.w      = rgen.normal(scale=0.01, size=X.shape[1] + 1)
           self.errors = []
           
           for i in self.n_iter:
               errors       = self.output(X) - y
               gradient     = X.T @ errors
               self.w[1:]  += gradient * self.eta
               self.w[0]   += errors.sum() * self.eta
               cost         = (errors**2).sum() / 2
               self.errors.append(cost)
               
class RidgeRegression(LinearRegression):  
    def __init__(self):
        super(RidgeRegression, self).__init__(standardize)
            
           