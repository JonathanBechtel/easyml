# -*- coding: utf-8 -*-
"""
Polynomial Regression, using standard OLS and Ridge Regression to 
"""
from utils import polynomialize
from linear_regression import LinearRegression
from ridge_regression import RidgeRegression

class PolynomialRegression(LinearRegression):
    def __init__(self, degree=2, n_iter=1000, eta=.001, centered=True, gd=True):
        self.degree = degree
        super().__init__(n_iter, eta, centered, gd)
    
    def fit(self, X, y):
        X = polynomialize(X)
        super().fit(X, y)
        
    def predict(self, X):
        X = polynomialize(X)
        super().predict(X)
        
class PolynomialRidgeRegression(RidgeRegression):
    def __init__(self, degree=2, n_iter=1000, eta=.001, centered=True, gd=True, alpha=1):
        self.degree = degree
        super().__init__(n_iter, eta, centered, gd, alpha)
    
    def fit(self, X, y):
        X = polynomialize(X)
        super().fit(X, y)