# -*- coding: utf-8 -*-
"""
Handcoded implementation of Linear and Logistic Regression, using Stochastic Gradient Descent as an optimization technique,
with the ability to use it with an L2 penalty
"""
import numpy as np
from utils import sigmoid

class SGDRegressor():
    """
    Class that initializes the Stochastic Gradient Regressor.
    
    PARAMETERS
        
    n_iter: int, default: 500
    The number of iterations to use in Gradient Descent
    ----------------------------------------------------
    
    eta: float, default: .01
    Learning rate used to update the gradient after each iteration
    ---------------------------------------------------
    
    w_initialized: boolean, default: False
    Boolean value that determines whether or not weights are initialized or not
    ---------------------------------------------------
    
    shuffle: boolean, default: True
    Whether or not you should shuffle indices of input data before fitting
    ---------------------------------------------------
    
    alpha: int, default: 0
    Determines strength of regularization.  Set to 0, which will give the same results
    as standard OLS
    --------------------------------------------------
    
    ATTRIBUTES
    
    w_: 1D array
    Weights after fitting
    --------------------------------------------------
    
    cost_: list
    Sum of squares cost function, with regularization term averaged over all training samples in each epoch
    """
    def __init__(self, n_iter=500, eta=.01, w_initialized=False, shuffle=True, alpha=0):
        self.n_iter        = 500
        self.eta           = .01
        self.w_initialized = w_initialized
        self.shuffle       = shuffle
        self.alpha         = alpha

    def fit(self, X, y):
        """
        Fit the training data
        """
        
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """
        Fit training data without re-initializing the weights
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
            
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        
        return self
    
    def _shuffle(self, X, y):
        """
        Randomly shuffle indices for X and y
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        self.w_            = np.random.normal(0, 0.1, size=m+1)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """
        Apply Gradient Descent to Update Weights of Variables
        """
        output       = self.activation(self.net_input(xi))
        error        = target - output
        self.w_[1:] += self.eta * xi @ error + self.regularization_grad
        self.w_[0]  += self.eta * error
        cost         = 0.5 * (error**2 + self.regularization_term)
        return cost
    
    def net_input(self, X):
        """
        Linear output function, y = mx + b
        """
        return X @ self.w_[1:] + self.w_[0]
    
    def activation(self, X):
        """
        Activation Function, if needed
        """
        return X
    
    def predict(self, X):
        """
        Predict Target Variable based on X
        """
        return self.net_input(X)
    
    @property
    def regularization_term(self):
        """
        L2 regularization, to be used for updating cost
        """
        return self.alpha * (self.w_[1:] @ self.w_[1:])
    
    @property
    def regularization_grad(self):
        """
        L2 gradient, to be used in update rule for weights
        """
        return self.alpha * self.w_[1:]
    
class SGDClassifier(SGDRegressor):
    """
    Uses SGD for Classification
    Inherits from SGDRegressor, with changes made to activation function
    and prediction function to return probabilities
    """
    
    def activation(self, X):
        return sigmoid(X)
    
    def predict(self, X):
        np.where(self.activation(self.net_input(X)) > 0.50, 1, 0)