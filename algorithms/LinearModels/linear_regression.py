"""
Linear Regression, implemented in Numpy, using both the analytical and gradient descent based solutions
"""
import numpy as np
from utils import standardize

class LinearRegression():
    """
    Class that implements Linear Regression.
    Inputs:
        
    n_iter:    integer
    -----------------------
    Number of iterations to use in gradient descent
    
    eta:       float
    -----------------------
    Size of the learning rate.  Typically will be between 0.1 and .001
    
    centered:  bool
    -----------------------
    If True, automatically centers the data
    
    gd:        bool
    -----------------------
    Determines whether or not you use gradient descent.  If True, gradient descent 
    will be used to solve.  If not, closed form will be used (if possible).
    """
    
    def __init__(self, n_iter=500, eta=.001, centered=True, gd=True):
        self.n_iter   = n_iter
        self.eta      = eta
        self.centered = centered
        self.gd       = gd
        
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
        if self.centered:
            X_fit = standardize(X)
        else:
            X_fit = X
            
        # if gradient descent, then solve w/ closed form solution    
        if not self.gd:
            # add bias unit
            X_fit      = np.c_[np.ones(len(X_fit)), X_fit]
            self.coef_ = np.linalg.inv(X_fit.T @ X_fit) @ X_fit.T @ y
            
        # otherwise, use gradient descent
        else:
            rgen = np.random.RandomState()
            # initialize weights, adding an extra for the intercept
            self.coef_  = rgen.normal(loc=0, scale=0.1, size=X_fit.shape[1] + 1)
            self.cost_  = []
            
            for i in range(self.n_iter):
                output          = self.predict(X_fit)           # create prediction
                errors          = y - output                    # get errors
                gradient        = X_fit.T @ errors * 1/len(X)   # get gradient w.r.t. each column, scale by # of samples
                self.coef_[1:] += gradient * self.eta           # update weights
                self.coef_[0]  += errors.sum() * self.eta * 1/len(X) # update intercept -- no regularization
                cost            = np.sum(errors**2) / 2         # calculate cost
                self.cost_.append(cost)                         # log it