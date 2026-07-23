#!/usr/bin/env python3
"""function train_tree(clf, X, y) to train a tree-based classifier using Scikit-learn.

Arguments:

clf: A Scikit-learn classifier instance
X: Input features
y: Target labels
Returns:

None
"""

import
from sklearn.tree import DecisionTreeClassifier


def train_tree(clf, X, y):
    clf.fit(X, y)
