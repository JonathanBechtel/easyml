"""
Ridge Regression, coded in Numpy, using either the closed form solution or Gradient Descent
"""
import numpy as np
import sys
sys.path.append('../..')
from utils import standardize

class RidgeRegression():
    
    def __init__(self, n_iter=500, eta=.001, standardize=True, gd=True, alpha=1):
        self.n_iter      = n_iter
        self.eta         = eta
        self.standardize = standardize
        self.gd          = gd
        self.alpha       = alpha
        
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
            self.w = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T @ y
            
        # otherwise, use gradient descent
        else:
            rgen = np.random.RandomState()
            # initialize weights, adding an extra for the intercept
            self.w      = rgen.normal(loc=0, scale=0.1, size=X.shape[1] + 1)
            self.cost_  = []
            
            for i in range(self.n_iter):
                l2_grad     = self.alpha * self.w[1:]
                l2_penalty  = self.alpha * np.sum(self.w[1:]**2)
                output      = self.predict(X)      
                errors      = y - output           
                gradient    = X.T @ errors + l2_grad
                self.w[1:] += gradient * self.eta * (1 / len(X))
                self.w[0]  += errors.sum() * self.eta
                cost        = (np.sum(errors**2) + l2_penalty) / 2
                self.cost_.append(cost)