# -*- coding: utf-8 -*-
"""
Lasso Regression Implemented with Coordinate Descent, written in Numpy
"""
import numpy as np

class LassoRegression():
    
    def __init__(self, n_iter=500, alpha=1):
        self.n_iter = n_iter
        self.alpha  = alpha
        
    def compute_rho(self, X, y, col_idx):
        pred   = self.predict(X)
        errors = y - pred
        rho    = X[:, col_idx] @ (errors + self.coef_[col_idx]*X[:, col_idx])
        return rho
    
    def predict(self, X):
        return X @ self.coef_
    
    def compute_zeta(self, X, col_idx):
        return  1 / np.sum(X[:, col_idx]**2)
    
    def soft_threshold(self, rho, alpha):
        if rho < -alpha:
            return rho + alpha
        elif rho >= -alpha and rho <= alpha:
            return 0
        else:
            return rho - alpha
        
    def fit(self, X, y):
        self.coef_  = np.random.normal(loc=0.0, scale=0.1, size=X.shape[1])
        
        for _ in range(self.n_iter):
            for col in range(len(self.coef_)):
                rho  = self.compute_rho(X, y, col)
                zeta = self.compute_zeta(X, col) 
                self.coef_[col] = zeta * self.soft_threshold(rho, self.alpha)