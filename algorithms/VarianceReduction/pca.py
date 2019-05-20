"""
Created on Mon May 20 13:34:54 2019

Handwritten version of Principal Component Analysis, written in Numpy

"""
import numpy as np
from utils import standardize

class PCA():
    """
    Class that performs Principal Component Analysis
    Values provided at input:
        
    n_components: integer
    ---------------------
    The number of components to be returned after the transformation.
    Doesn't specify a default, but will set it to the number of columns
    provided in dataset when the fit() method is called
    
    standardize:  boolean
    ---------------------
    Whether or not to standardize the dataset passed in.  Default value is True.
    """
    def __init__(self, n_components=None, standardize=True):
        """Initialize the PCA algorithm"""
        self.n_components = n_components
        self.standardize  = standardize
        
    def fit(self, X):
        """Determine the eigenvalues and eigenvectors of the feature matrix, sorts them according to 
        absolute value of the returned eigenvalues.  Function returns itself to be chained w/ the
        fit_transform() method"""
        
        # if no value for n_components is specified, create one for each column in dataset 
        if self.n_components is None:
            self.n_components  = X.shape[1]
        
        # standardize dataset, if specified
        if self.standardize:
            X = standardize(X)
        
        # create covariance matrix, perform eigen decomposition
        # return the eigenvalues and eigen vectors from decomposition
        cov_mat                = np.cov(X.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        
        # sort the eigen values from high to low
        self.variance_ratios_       = sorted([ (eigen_vals[i] / np.sum(eigen_vals)) for i in range(self.n_components)], reverse=True)
        # pair each eigen value with its eigen vector
        self.eigen_pairs            = [(eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        # sort from high to low
        self.sorted_pairs           = sorted(self.eigen_pairs, reverse=True)
        
        return self
        
    def transform(self, X):
        """
        Creates new feature matrix from eigen vectors and original feature matrix
        """
        
        # stack components together in a new numpy array
        components = np.hstack((self.sorted_pairs[i][1][:, np.newaxis] for i in range(self.n_components)))
        # take the dot product with X, to be used for later analysis
        X_pca      = X @ components
        
        return X_pca
    
    def fit_transform(self, X):
        """Chains together fir and transform methods to create new version of X in one function"""
        return self.fit(X).transform(X)