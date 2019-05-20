"""
Created on Mon May 20 13:34:54 2019

@author: Jonat
"""
import numpy as np
from utils import standardize

class PCA():
    def __init__(self, n_components=None, standardize=True):
        self.n_components = n_components
        self.standardize  = standardize
        
    def fit(self, X):
        
        if self.n_components is None:
            self.n_components  = X.shape[1]
        
        if self.standardize:
            X = standardize(X)
        
        cov_mat                = np.cov(X.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        
        self.variance_ratios_       = sorted([ (eigen_vals[i] / np.sum(eigen_vals)) for i in range(self.n_components)], reverse=True)
        self.eigen_pairs            = [(eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        self.sorted_pairs           = sorted(self.eigen_pairs, reverse=True)
        
        return self
        
    def transform(self, X):
        w     = np.hstack((self.sorted_pairs[i][1][:, np.newaxis] for i in range(self.n_components)))
        X_pca = X @ w
        
        return X_pca
    
    def fit_transform(self, X):
        
        return self.fit(X).transform(X)