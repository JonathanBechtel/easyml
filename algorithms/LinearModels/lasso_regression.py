"""
Lasso Regression Implemented with Coordinate Descent, written in Numpy
"""
import numpy as np
from utils import standardize

class LassoRegression():
    """
    Takes the following variables at initialization:
    n_iter: number of iterations to use for gradient descent
    alpha:  value of alpha to use for regularization
    """
    def __init__(self, n_iter=500, alpha=1, centered=True):
        self.n_iter   = n_iter
        self.alpha    = alpha
        self.centered = centered
        
    def compute_rho(self, X, y, col_idx):
        """sub derivative of variable"""
        pred   = self.predict(X)
        errors = y - pred
        rho    = X[:, col_idx] @ (errors + self.coef_[col_idx]*X[:, col_idx])
        return rho
    
    def predict(self, X):
        return X @ self.coef_
    
    def compute_zeta(self, X, col_idx):
        """Value of zeta, used in the coordinate descent update"""
        return  1 / np.sum(X[:, col_idx]**2)
    
    def soft_threshold(self, rho, alpha):
        """Soft threshold function to determine value of coefficient"""
        if rho < -alpha:
            return rho + alpha
        elif rho >= -alpha and rho <= alpha:
            return 0
        else:
            return rho - alpha
        
    def fit(self, X, y):
        
        if self.centered:
            X_pred = standardize(X)
        else:
            X_pred = X
            
        # update weights through coordinate descent
        self.coef_  = np.random.normal(loc=0.0, scale=0.1, size=X_pred.shape[1])
        
        for _ in range(self.n_iter):
            for col in range(len(self.coef_)):
                rho  = self.compute_rho(X_pred, y, col)
                zeta = self.compute_zeta(X_pred, col) 
                self.coef_[col] = zeta * self.soft_threshold(rho, self.alpha)