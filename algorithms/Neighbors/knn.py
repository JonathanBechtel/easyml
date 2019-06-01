# -*- coding: utf-8 -*-
"""
Handwritten version of K-Nearest Neighbors, written in Numpy
"""
import numpy as np
from utils import standardize, majority_vote

class KNN():
    """
    Class that implements the K-Nearest Neighbors Algorithm
    Takes the following inputs at initiation:
        
    neighbors: int
    --------------------
    Number of neighbors to use when making a sample prediction
    
    centered: bool
    --------------------
    Whether or not to standardize the data when fitting
    """
    
    def __init__(self, neighbors=5, centered=True):
        self.neighbors = neighbors
        self.centered  = centered
        
    def _get_distance(self, xi):
        """
        Calculates the Euclidean Distance between a given point and every other one inside a matrix
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
        # create matrix that for each sample in X holds a distance calculation for every sample in X_fit
        m, n              = X.shape[0], self.X_fit.shape[0]
        self.dist_matrix  = np.zeros((m, n))
        X_pred            = np.zeros(X.shape)
        
        # if specified, standardize the data
        if standardize:
           X_pred  = standardize(X)
        else:
           X_pred  = X
        
        # for each row in X, get its euclidean distance from every member in X_fit
        for row in range(m):
            self.dist_matrix[row] = self._get_distance(X_pred[row])
            
        # for each sample, return the indices of the K nearest neighbors    
        self.idx_vals      = np.argsort(self.dist_matrix)[:, :self.neighbors]
        # find corresponding values in y for each neighbor
        self.y_idx         = self.y_fit[self.idx_vals]
        # perform appropriate calculation member of y
        preds         = [self.neighbor_calculation(self.y_idx[i]) for i in range(len(self.y_idx))]
        return preds

class KNNClassifier(KNN):
    """
    Subclass of KNN, used for classification.  
    Sets calculation method for each neighbor to a majority vote
    """
    def predict(self, X, centered=False):
        self.neighbor_calculation = majority_vote
        return super().predict(X, centered)
        
class KNNRegressor(KNN):
    """
    Subclass of KNN, used for regression
    Sets calculation method for each neighbor to be the average
    value of all neighbors
    """
    def predict(self, X, centered=False):
        self.neighbor_calculation = np.mean
        return super().predict(X, centered)        