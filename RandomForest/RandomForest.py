# Import necessary libraries
from collections import Counter
import numpy as np
# Import DecisionTree class from a custom module
from DecisionTrees.DecisionTree import DecisionTree
import sys
import os

# Add the parent directory to the Python path to import custom modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


class RandomForest:
    """
    A class to implement a Random Forest algorithm.

    Attributes:
    n_trees (int): The number of decision trees to combine.
    max_depth (int): The maximum depth of each decision tree.
    min_samples_split (int): The minimum number of samples required to split a node.
    n_features (int): The number of features to consider at each split.
    trees (list): A list to store the decision trees.
    """

    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        """
        Initialize the Random Forest object.

        Parameters:
        n_trees (int): The number of decision trees to combine (default=10).
        max_depth (int): The maximum depth of each decision tree (default=10).
        min_samples_split (int): The minimum number of samples required to split a node (default=2).
        n_features (int): The number of features to consider at each split (default=None).
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        """
        Train the Random Forest model on the given data.

        Parameters:
        X (numpy array): The feature matrix.
        y (numpy array): The target vector.
        """
        self.trees = []
        for _ in range(self.n_trees):
            # Create a new decision tree with the given parameters
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)

            # Generate bootstrap samples from the training data
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # Train the decision tree on the bootstrap samples
            tree.fit(X_sample, y_sample)

            # Add the trained tree to the list of trees
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        """
        Generate bootstrap samples from the given data.

        Parameters:
        X (numpy array): The feature matrix.
        y (numpy array): The target vector.

        Returns:
        X_sample (numpy array): The sampled feature matrix.
        y_sample (numpy array): The sampled target vector.
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        """
        Find the most common label in the given array.

        Parameters:
        y (numpy array): The target vector.

        Returns:
        most_common (int): The most common label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Make predictions on the given data.

        Parameters:
        X (numpy array): The feature matrix.

        Returns:
        predictions (numpy array): The predicted labels.
        """
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Swap axes to get predictions for each sample
        tree_preds = np.swapaxes(predictions, 0, 1)

        # Find the most common label for each sample
        predictions = np.array([self._most_common_label(pred)
                               for pred in tree_preds])

        return predictions
