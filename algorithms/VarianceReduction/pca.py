"""
Handwritten version of Principal Component Analysis, written in Numpy
"""
import numpy as np
from utils import standardize

class PCA():
    """
    Class that performs Principal Component Analysis
    Values provided at input:
        
    n_components: integer || default: None
    ---------------------
    The number of components to be returned after the transformation.
    Doesn't specify a default, but will set it to the number of columns
    provided in dataset when the fit() method is called
    
    centered:  boolean || default: True
    ---------------------
    Whether or not to standardize the dataset passed in.
    
    ATTRIBUTES:
        
    variance_ratios_: list
    -----------------------------
    The amount of explained variance for each component, listed in descending order.  Available
    after calling fit()
    
    components_: 2D array
    -----------------------------
    2D array consisting of the eigenvectors used to transform dataset.  # of rows corresponds to # of samples
    in dataset.  # of columns corresponds to # of components specified at initialization.
    """
    def __init__(self, n_components=None, centered=True):
        """Initialize the PCA algorithm"""
        self.n_components = n_components
        self.centered     = centered
        
    def fit(self, X):
        """
        Determine the eigenvalues and eigenvectors of the feature matrix.
        Returns itself to be chained w/ the fit_transform() method
        """
        
        # if no value for n_components is specified, create one for each column in dataset 
        if self.n_components is None:
            self.n_components  = X.shape[1]
        
        # standardize dataset, if specified
        if self.centered:
            X = standardize(X)
        
        # create covariance matrix, perform eigen decomposition
        # return the eigenvalues and eigen vectors from decomposition
        cov_mat                = np.cov(X.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        
        # sort the eigen values from high to low
        self.variance_ratios_  = sorted([ (eigen_vals[i] / np.sum(eigen_vals)) for i in range(self.n_components)], reverse=True)
        # pair each eigen value with its eigen vector
        eigen_pairs = [(eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        # sort from high to low
        sorted_pairs = sorted(eigen_pairs, reverse=True)
        # stack components in appropriate order
        self.components_ = np.hstack((sorted_pairs[i][1][:, np.newaxis] for i in range(self.n_components)))
        
        return self
        
    def transform(self, X):
        """
        Creates new feature matrix from eigen vectors and original feature matrix
        """
        X_pca      = X @ self.components_
        
        return X_pca
    
    def fit_transform(self, X):
        """
        Chains together fit and transform methods to create new version of X in one function
        """
        return self.fit(X).transform(X)