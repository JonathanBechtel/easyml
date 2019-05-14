# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:28:32 2019

@author: Jonat
"""

import numpy as np
from utils import mse, gini, majority_vote

class TreeNode():
    """
    Class that represents a decision point for each split in the tree.  Determines whether or not you continue to traverse the tree, or stop at a leaf and calculate
    its value.
    """
    def __init__(self, feature=None, feature_value=None, leaf_value=None, branch_size=None, left_branch=None, right_branch=None):
        self.feature       = feature,
        self.feature_value = feature_value
        self.leaf_value    = leaf_value
        self.branch_size   = branch_size
        self.left_branch   = left_branch
        self.right_branch  = right_branch

class DecisionTree():
    
    def __init__(self, min_leaf=5, impurity_threshold=1e-5, root=None, leaf_value=None, impurity=None):
        
        self.min_leaf              = min_leaf
        self.impurity_threshold    = impurity_threshold
        self.root                  = root
        self._leaf_calculation     = leaf_value
        self._impurity_calculation = impurity
        
    def _get_splits(self, X, y, col_idx, value):
        
        lhs = np.nonzero(X[:, col_idx] <= value)
        rhs = np.nonzero(X[:, col_idx] >  value)
        
        return lhs[0], rhs[0]
    
    def _info_gain(self, X, y, col_idx, value):
        # get left and right splits for given value
        lhs, rhs = self._get_splits(X, y, col_idx, value)
        
        if len(rhs) is 0 or len(lhs) is 0:
            return 0
        
        #calculate relative weight of each split depending on sample size
        sample_size  = len(X)
        left_weight  = len(lhs) / sample_size
        right_weight = len(rhs) / sample_size
         
        # calculate information gain from this particular split
        info_change = self._impurity_calculation(y, y.mean()) - (left_weight * self._impurity_calculation(y[lhs], y[lhs].mean())) - (right_weight * self._impurity_calculation(y[rhs], y[rhs].mean()))
        
        return info_change
    
    def _find_best_value(self, X, y, col_idx):
        # generate unique values
        unique_values = np.unique(X[:, col_idx])
        
        # return the value that gives largest amt of info gain among unique values
        best_val = max((self._info_gain(X, y, col_idx, value), value) for value in unique_values)
        
        return best_val
    
    def _find_best_split(self, X, y):
        # get the best value in the dataset
        best_value = max((self._find_best_value(X, y, i), i) for i in range(X.shape[1]))  
    
        # unpack it into the info_change, column value, and column # for the best split
        info_change, value, col_number = best_value[0][0], best_value[0][1], best_value[1]
    
        # return unpacked values
        return info_change, value, col_number
    
    def _is_leaf(self, info_change, split):
        """Determines if a decision node represents a leaf or not"""
        if info_change <= self.impurity_threshold:
            return True
        if len(split) <= self.min_leaf:
            return True
        else:
            return False
    
    def _build_tree(self, X, y):

        # get info gain, column value and column number of best split
        info_change, value, col_number = self._find_best_split(X, y)
        
        # generate left and right sides of tree based on best value
        lhs, rhs = self._get_splits(X, y, col_number, value)
        
        # if both splits are leaves, return them as such
        if self._is_leaf(info_change, lhs) or self._is_leaf(info_change, rhs):
            return TreeNode(branch_size = len(X),
                            leaf_value  = self._leaf_calculation(y))
        
        else:
            return TreeNode(feature       = col_number, 
                            feature_value = value, 
                            branch_size   = len(X), 
                            left_branch   = self._build_tree(X[lhs], y[lhs]),
                            right_branch  = self._build_tree(X[rhs], y[rhs]))
            

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _find_leaf(self, x_i, tree=None):
        
        if tree is None:
            tree = self.root
            return self._find_leaf(x_i, tree)
            
        if tree.leaf_value is not None:
            return tree.leaf_value
        else:
            if x_i[tree.feature[0]] <= tree.feature_value:
                return self._find_leaf(x_i, tree.left_branch)
            else:
                return self._find_leaf(x_i, tree.right_branch)
            
    def predict(self, X):
        preds = [self._find_leaf(x_i) for x_i in X]
        return preds
            
    def print_tree(self, tree=None):
        """ Recursively print decision tree """
        if not tree:
            tree = self.root
            self.print_tree(tree=tree)
    
            # If we're at leaf => print the label
        if tree.leaf_value is not None:
            print ("Leaf: ", tree.leaf_value, "Leaf Size: ", tree.branch_size)
                
            # Go deeper down the tree
        else:
            # Print branch value
            print ("%s: %s " % (tree.feature[0], tree.feature_value))
            # Left branch
            print ("Left->", end=" ")
            self.print_tree(tree.left_branch)
            # Right Branch
            print ("Right->", end=" ")
            self.print_tree(tree.right_branch)
            
class DecisionTreeRegressor(DecisionTree):
    
    def fit(self, X, y):
        self._leaf_calculation = np.mean
        self._impurity_calculation = mse
        super().fit(X, y)

    
class DecisionTreeClassifier(DecisionTree):
    
    def fit(self, X, y):
        self._leaf_calculation = majority_vote
        self._impurity_calculation = gini
        super().fit(X, y)