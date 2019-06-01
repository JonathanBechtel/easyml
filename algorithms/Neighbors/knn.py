# -*- coding: utf-8 -*-
"""
Handwritten version of K-Nearest Neighbors, written in Numpy
"""
import numpy as np
from utils import standardize, majority_vote

class KNN():
    
    def __init__(self, neighbors=5, centered=True):
        self.neighbors = neighbors
        self.centered  = centered
        
    def _get_distance(self, xi):
        """
        Calculate the Euclidean Distance between a given point and every other one inside a matrix
        """
        return np.sqrt(((xi - self.X_fit)**2).sum(1))
        
    def fit(self, X, y):
        """
        Stores given data to be used for calculating distance in predict method
        """
        
        # standardize data if you need to
        if self.centered:
            self.X_fit = standardize(X)
        # store values to be used for prediction
        else:
            self.X_fit = X
        self.y_fit     = y

    def predict(self, X, centered=False):
        """
        Predicts value of sample based on computed distance matrix and value calculation
        """
        m, n              = X.shape[0], self.X_fit.shape[0]
        self.dist_matrix  = np.zeros((m, n))
        X_pred            = np.zeros(X.shape)
        
        if standardize:
           X_pred  = standardize(X)
        else:
           X_pred  = X
        
        for row in range(m):
            self.dist_matrix[row] = self._get_distance(X_pred[row])
            
        self.idx_vals      = np.argsort(self.dist_matrix)[:, :self.neighbors]
        self.y_idx         = self.y_fit[self.idx_vals]
        self.preds              = [self.neighbor_calculation(self.y_idx[i]) for i in range(len(self.y_idx))]
        return self.preds

class KNNClassifier(KNN):
    def predict(self, X, centered=False):
        self.neighbor_calculation = majority_vote
        return super().predict(X, centered)
        
class KNNRegressor(KNN):
    def predict(self, X, centered=False):
        self.neighbor_calculation = np.mean
        return super().predict(X, centered)        