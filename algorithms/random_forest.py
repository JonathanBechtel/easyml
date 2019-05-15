# -*- coding: utf-8 -*-
"""
Handcoded Random Forest for Regression and Classification written in Numpy.
Creates a bases class RandomForest, which is then passed onto RandomForestClassifier and RandomForestRegressor, 
each to be used for the appropriate use.
"""
import numpy as np
from decision_tree import DecisionTreeRegressor, DecisionTreeClassifier
from utils import shuffle_matrix, majority_vote

class RandomForest():
    def __init__(self, n_estimators=10, min_leaf_size=5, sample_size = 2/3, min_impurity=1e-5):
        self.n_estimators = n_estimators
        self.min_leaf     = min_leaf_size
        self.sample_size  = sample_size
        self.min_impurity = min_impurity
    
    def fit(self, X, y):
        Xy             = np.c_[X, y]
        for tree in self.trees:
            Xy_sample  = shuffle_matrix(Xy, sample_size=self.sample_size)
            X_new      = Xy_sample[:, :-1]
            y_new      = Xy_sample[:, -1]
            tree.fit(X_new, y_new)
            
    def predict(self, X):
        y_pred = [self._tree_calculation([tree._find_leaf(x_i) for tree in self.trees]) for x_i in X]
        return y_pred
    
class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=10, min_leaf_size=5, sample_size = 2/3, min_impurity=1e-5):
        super().__init__(n_estimators, min_leaf_size, sample_size, min_impurity) 
        self.tree  = DecisionTreeRegressor
        self._tree_calculation = np.mean
        self.trees = [self.tree(min_leaf=self.min_leaf, impurity_threshold=self.min_impurity) for i in range(self.n_estimators)]
        
        
class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, min_leaf_size=5, sample_size=2/3, min_impurity=1e-5):
        super().__init__(n_estimators, min_leaf_size, sample_size, min_impurity)
        self.tree  = DecisionTreeClassifier
        self._tree_calculation = majority_vote
        self.trees = [self.tree(min_leaf=self.min_leaf, impurity_threshold=self.min_impurity) for i in range(self.n_estimators)]