"""
<<<<<<< HEAD
Handcoded Decision Tree, which is later inherited by child classes to be used for Classification and Regression.
=======
Handspun implementation of a DecisionTree, with inherited classes to use for Regression and Classification, written in Numpy.
>>>>>>> a11b13987ab09d93d1de9d82fbd2f8baef2dc48d
"""

import numpy as np
import sys
sys.path.append('..')
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
    """
    Base Decision Tree Class used for Regression and Classification Trees
    
    Takes the following inputs:
    
    min_leaf:            minimum number of samples you can have on a leaf in order to create a split
    impurity_threshold:  minimum amount of information gain required to create a split
    root:                the split that represents the base of your tree
    """
    
    def __init__(self, min_leaf=5, impurity_threshold=1e-5, root=None):
        
        self.min_leaf              = min_leaf
        self.impurity_threshold    = impurity_threshold
        self.root                  = root
        
    def _get_splits(self, X, y, col_idx, value):
        """
        Creates indices for left and right splits to be used in building the tree
        """
        lhs = np.nonzero(X[:, col_idx] <= value)
        rhs = np.nonzero(X[:, col_idx] >  value)
        
        return lhs[0], rhs[0]
    
    def _info_gain(self, X, y, col_idx, value):
        """
        Determines amount of info_gain from splitting your dataset with a particular value in a particular column
        """
        
        # get left and right splits for given value
        lhs, rhs = self._get_splits(X, y, col_idx, value)
        
        # no info gain if either split has no samples
        if len(rhs) is 0 or len(lhs) is 0:
            return 0
        
        #calculate relative weight of each split depending on sample size
        sample_size  = len(X)
        left_weight  = len(lhs) / sample_size
        right_weight = len(rhs) / sample_size
         
        # calculate information gain from this particular split - self._impurity_calculation is determined in subclasses
        info_change = self._impurity_calculation(y, y.mean()) - (left_weight * self._impurity_calculation(y[lhs], y[lhs].mean())) - (right_weight * self._impurity_calculation(y[rhs], y[rhs].mean()))
        
        return info_change
    
    def _find_best_value(self, X, y, col_idx):
        """
        For any given column, finds the value that returns the largest amount of information gain
        """
        # generate unique values
        unique_values = np.unique(X[:, col_idx])
        
        # return the value that gives largest amt of info gain among unique values
        best_val = max((self._info_gain(X, y, col_idx, value), value) for value in unique_values)
        
        return best_val
    
    def _find_best_split(self, X, y):
        """
        For a feature matrix and target variable, iterates through each column to find the best value to split on
        """
        # get the best value in the dataset
        best_value = max((self._find_best_value(X, y, i), i) for i in range(X.shape[1]))  
    
        # unpack it into the info_change, column value, and column # for the best split
        info_change, value, col_number = best_value[0][0], best_value[0][1], best_value[1]
    
        # return unpacked values
        return info_change, value, col_number
    
    def _is_leaf(self, info_change, split):
        """
        Determines if a decision node represents a leaf or not
        """
        
        if info_change <= self.impurity_threshold:
            return True
        if len(split) <= self.min_leaf:
            return True
        else:
            return False
    
    def _build_tree(self, X, y):
        """
        Recursively finds the best split, determines point in tree, and either calculates a leaf value or splits again
        """
        # get info gain, column value and column number of best split
        info_change, value, col_number = self._find_best_split(X, y)
        
        # generate left and right sides of tree based on best value
        lhs, rhs = self._get_splits(X, y, col_number, value)
        
        # if either split is a leaf, then create a leaf
        if self._is_leaf(info_change, lhs) or self._is_leaf(info_change, rhs):
            return TreeNode(branch_size = len(X),
                            leaf_value  = self._leaf_calculation(y))
        
        # if not, then keep on splitting
        else:
            return TreeNode(feature       = col_number, 
                            feature_value = value, 
                            branch_size   = len(X), 
                            left_branch   = self._build_tree(X[lhs], y[lhs]),
                            right_branch  = self._build_tree(X[rhs], y[rhs]))
            

    def fit(self, X, y):
        """
        Fits the dataset to a decision tree by calling the _build_tree function
        """
        self.root = self._build_tree(X, y)
        
    def _find_leaf(self, x_i, tree=None):
        """
        Takes a sample, recursively travels down a tree until it ends up at a leaf, where its value is calculated
        """
        
        # if no tree, start at its root
        if tree is None:
            tree = self.root
            return self._find_leaf(x_i, tree)
            
       # if on a leaf, then return the leaf value 
        if tree.leaf_value is not None:
            return tree.leaf_value
        # if not, then determine what branch you want to travel down
        else:
            if x_i[tree.feature[0]] <= tree.feature_value:
                return self._find_leaf(x_i, tree.left_branch)
            else:
                return self._find_leaf(x_i, tree.right_branch)
            
    def predict(self, X):
        """
        Builds predictions for input matrix by assigning each row to a leaf
        """
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
    """
    Inherits DecisionTree class, sets leaf_calculation and impurity calculation to make it suitable for regression
    """
    def fit(self, X, y):
        self._leaf_calculation = np.mean
        self._impurity_calculation = mse
        super().fit(X, y)

    
class DecisionTreeClassifier(DecisionTree):
    """
    Inherits DecisionTree class, sets leaf_calculation and impurity calculation to make it suitable for classification
    """  
    def fit(self, X, y):
        self._leaf_calculation = majority_vote
        self._impurity_calculation = gini
        super().fit(X, y)
