"""
Handwritten version of Linear Discriminant Analysis, written in Numpy
"""
import numpy as np
from utils import standardize

class LDA():
    """
    Class that implements linear discriminant analysis, a variance reduction technique that
    finds linear combinations of features that best describe a dataset.  
    
    Values provided at input:
        
    n_discriminants: integer || default: None
    ----------------------------
    The number of features to separate your dataset on.  If default is kept, no columns will be removed
    from your dataset.  If a value is provided, this is the number of discriminants(features) that will
    be used to separate your data.
    
    centered: boolean || default: False
    -----------------------------
    Whether or not to standardize your data before fitting.  If set to true, every column in feature matrix
    will be standardized to have a mean of 0 and variance of 1.
    
    ATTRIBUTES:
        
    variance_ratios_: list
    -----------------------------
    The amount of explained variance for each discriminant, listed in descending order.  Available
    after calling fit()
    
    discriminants_: 2D array
    -----------------------------
    2D array consisting of the eigenvectors used to transform dataset.  # of rows corresponds to # of samples
    in dataset.  # of columns corresponds to # of discriminants specified at initialization.
    """
    
    def __init__(self, n_discriminants=None, centered=False):
        self.n_discriminants = n_discriminants
        self.centered        = centered
        
    def scatter_matrix_col(self, X, y, val):
        """Finds column average when y has a particular value"""
        matrix_col = X[y == val].mean(0)
        return matrix_col
    
    def build_scatter_matrix(self, X, y):
        """
        Uses the scatter_matrix_col() method on each unique class value of y, and
        stacks them together in an ndarray
        """
        y_vals         = np.unique(y)
        scatter_matrix = np.hstack((self.scatter_matrix_col(X, y, val)[:, np.newaxis] for val in y_vals))
        return scatter_matrix
        
    def within_class_matrix(self, X, y):
        """
        Returns an mxm scatter matrix that represents variance within a feature
        with respect to each of its unique values
        """
        m_features     = X.shape[1]
        y_vals         = np.unique(y)
        S_w            = np.zeros((m_features, m_features))
        for val in y_vals:
            scat_matrix = np.cov(X[y == val].T)
            S_w         += scat_matrix
        return S_w
        
    def between_class_matrix(self, X, y):
        """
        Returns an mxm scatter matrix that represents the variance between columns
        with respect to each unique value of y
        """
        col_means  = X.mean(0)[:, np.newaxis]
        mean_vecs  = self.build_scatter_matrix(X, y)
        y_vals     = np.unique(y)
        m_features = X.shape[1]
        # empty matrix that will be updated
        S_b        = np.zeros((m_features, m_features))
        
        # for each unique value of y
        for i, val in enumerate(y_vals):
            # count how many times it occurs
            n           = np.sum(y == val)
            # get the difference between average value for each column
            # when this value of y occurs, and average value for the
            # column as a whole
            val         = mean_vecs[:, i][:, np.newaxis] - col_means
            # get the dot product for this value
            scat_matrix = val @ val.T * n
            # update the scatter matrix
            S_b         += scat_matrix
        return S_b
    
    def fit(self, X, y):
        """Transforms dataset using computed S_w & S_b matrices to form new discriminants"""
        
        # if number of discriminants is not specified, make it equal to # of columns in dataset
        if self.n_discriminants is None:
            self.n_discriminants  = X.shape[1]
            
        # standardize data if specified    
        if self.centered:
            X_fit = standardize(X)
        else:
            X_fit = X
            
        # calculate S_w and S_b, to be used for eigen decomposition    
        S_b    = self.between_class_matrix(X_fit, y)
        S_w    = self.within_class_matrix(X_fit, y)
        inv_Sw = np.linalg.inv(S_w)
        
        # get eigen values and eigen vectors to be used for data transformation
        eigen_vals, eigen_vecs = np.linalg.eig(inv_Sw @ S_b)
        
        # pair each eigen value with its eigen vector
        eigen_pairs           = [(eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        # sort from high to low
        sorted_pairs          = sorted(eigen_pairs, key=lambda x: x[0], reverse=True)
        # stack discriminants in appropriate order
        self.discriminants_   = np.hstack((sorted_pairs[i][1][:, np.newaxis].real for i in range(self.n_discriminants)))
        # calculated total explained variance for included discriminants
        self.variance_ratios_ = [np.abs(pair[0].real)/np.sum(eigen_vals.real) for pair in sorted_pairs[:self.n_discriminants]]
        
        return self
        
    def transform(self, X):
        """
        Creates new feature matrix from eigen vectors and original feature matrix
        """
        X_lda = X @ self.discriminants_
        
        return X_lda
    
    def fit_transform(self, X, y):
        """
        Chains together fit and transform methods to create new version of X in one function
        """
        return self.fit(X, y).transform(X)