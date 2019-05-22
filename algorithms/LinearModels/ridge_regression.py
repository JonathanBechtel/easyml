"""
Ridge Regression, coded in Numpy, using either the closed form solution or Gradient Descent
"""
import numpy as np
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
            # initialize weights, adding an extra for the intercept
            self.w      = np.random.normal(loc=0, scale=0.1, size=X.shape[1] + 1)
            self.cost_  = []
            
            for i in range(self.n_iter):
                l2_grad     = self.alpha * self.w[1:]                 # update l2 gradient
                l2_penalty  = self.alpha * np.sum(self.w[1:]**2)      # update l2 loss term
                output      = self.predict(X)                         # make prediction - linear output
                errors      = y - output                              # get error column
                gradient    = (X.T @ errors + l2_grad) * 1/len(X)     # get error wrt to each column, add l2, scale by 1/m
                self.w[1:] += gradient * self.eta                     # update the weights by gradients * learning rate
                self.w[0]  += errors.sum() * self.eta  * 1/len(X)     # update intercept by error column * learning rate * 1/m
                cost        = (np.sum(errors**2) + l2_penalty) / 2    # compute the cost
                self.cost_.append(cost)                               # log it