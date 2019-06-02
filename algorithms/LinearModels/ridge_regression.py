"""
Ridge Regression, coded in Numpy, using either the closed form solution or Gradient Descent
"""
import numpy as np
from utils import standardize

class RidgeRegression():
    """
    Initializes Ridge Regression Algorithm
    Inputs:
        
    n_iter: integer
    ---------------------
    Number of iterations to use when implementing coordinate descent
    
    eta:    float
    ---------------------
    Size of the learning rate.  Typically between 0.1 and .001
    
    alpha: integer
    ---------------------
    Strength of tuning parameter alpha to be used in regularization
    
    centered: bool
    ---------------------
    Whether or not to standardize your data when calling fit()
    
    gd:   bool
    ---------------------
    Determines whether or not you use gradient descent to derive coefficients.
    If true, gradient descent is used.  If false, closed form solution is used.
    """
    
    def __init__(self, n_iter=500, eta=.001, centered=True, gd=True, alpha=1):
        self.n_iter      = n_iter
        self.eta         = eta
        self.centered     = centered
        self.gd          = gd
        self.alpha       = alpha
        
    def predict(self, X):
        """
        Creates predicted value by multipltying the feature matrix by their weights, and adding the intercept term
        """
        return X @ (self.coef_[1:]) + self.coef_[0]
    
    def fit(self, X, y):
        """
        Determine statistical relationship between columns in X and target variable y
        """
        # standardize feature matrix if needed
        X_fit = np.zeros(X.shape)
        if self.centered:
            X_fit = standardize(X)
        else:
            X_fit = X
            
        # if gradient descent, then solve w/ closed form solution    
        if not self.gd:
            # add bias unit
            X_fit      = np.c_[np.ones(len(X_fit)), X_fit]
            self.coef_ = np.linalg.inv(X_fit.T @ X_fit + self.alpha * np.eye(X_fit.shape[1])) @ X_fit.T @ y
            
        # otherwise, use gradient descent
        else:
            # initialize weights, adding an extra for the intercept
            self.coef_  = np.random.normal(loc=0, scale=0.1, size=X.shape[1] + 1)
            self.cost_  = []
            
            for i in range(self.n_iter):
                l2_grad        = self.alpha * self.coef_[1:]             # update l2 gradient
                l2_penalty     = self.alpha * np.sum(self.coef_[1:]**2)  # update l2 loss term
                output         = self.predict(X_fit)                     # make prediction - linear output
                errors         = y - output                              # get error column
                gradient       = (X_fit.T @ errors + l2_grad) * 1/len(X) # get error wrt to each column, add l2, scale by 1/m
                self.coef_[1:] += gradient * self.eta                    # update the weights by gradients * learning rate
                self.coef_[0]  += errors.sum() * self.eta  * 1/len(X)    # update intercept by error column * learning rate * 1/m
                cost           = (np.sum(errors**2) + l2_penalty) / 2    # compute the cost
                self.cost_.append(cost)                                  # log it