"""
Lasso Regression Implemented with Coordinate Descent, written in Numpy
"""
import numpy as np
from utils import standardize

class LassoRegression():
    """
    Initializes LassoRegression Algorithm
    Inputs:
        
    n_iter: integer
    ---------------------
    Number of iterations to use when implementing coordinate descent
    
    alpha: inter
    ---------------------
    Strength of tuning parameter alpha to be used in regularization
    
    centered: bool
    ---------------------
    Whether or not to standardize your data when calling fit()
    """
    def __init__(self, n_iter=500, alpha=1, centered=True):
        self.n_iter   = n_iter
        self.alpha    = alpha
        self.centered = centered
        
    def _compute_rho(self, X, y, col_idx):
        """Subdifferential of variable"""
        
        # generate linear output from coefficients
        pred   = self.predict(X)
        # get gradient, based off of MSE
        errors = y - pred
        # take subdifferential of this particular column
        rho    = X[:, col_idx] @ (errors + self.coef_[col_idx]*X[:, col_idx])
        return rho
    
    def predict(self, X):
        """Linear output based off of X and derived coefficients"""
        return X @ self.coef_
    
    def _compute_zeta(self, X, col_idx):
        """Value of zeta, used in the coordinate descent update"""
        return  1 / np.sum(X[:, col_idx]**2)
    
    def _soft_threshold(self, rho, alpha):
        """Soft threshold function to determine value of coefficient"""
        if rho < -alpha:
            return rho + alpha
        elif rho >= -alpha and rho <= alpha:
            return 0
        else:
            return rho - alpha
        
    def fit(self, X, y):
        
        # standardize data, if specified
        X_fit = np.zeros(X.shape)
        if self.centered:
            X_fit = standardize(X)
        else:
            X_fit = X
        # insert column of ones for intercept    
        X_fit = np.insert(X_fit, 0, 1, axis=1)
            
        # initialize weights for each variable
        self.coef_ = np.random.normal(loc=0.0, scale=0.1, size=X_fit.shape[1])
        
        # initialize coordinate descent
        for _ in range(self.n_iter):
            # for every column except intercept
            for col in range(1, len(self.coef_)):
                # take its subdifferential
                rho  = self.compute_rho(X_fit, y, col)
                # calculate zeta
                zeta = self.compute_zeta(X_fit, col)
                # update weight depending on value of subdifferential and alpha
                self.coef_[col] = zeta * self.soft_threshold(rho, self.alpha)
        # after you've updated all weights, set the intercept -- should be y.mean() if data is centered        
        self.coef_[0] = y.mean() - X_fit[:, 1:].mean(0) @ self.coef_[1:].T