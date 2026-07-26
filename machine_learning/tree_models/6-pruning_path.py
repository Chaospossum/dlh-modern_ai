#!/usr/bin/env python3
"""Retrieve the cost-complexity pruning path for a decision tree."""

def get_pruning_path(clf, X, y):
    """Retrieve the pruning path for a decision tree classifier.

    Args:
        clf: A trained DecisionTreeClassifier instance.
        X: Input features.
        y: Target labels.

    Returns:
        tuple: (ccp_alphas, impurities) as NumPy arrays.
    """
    return clf.cost_complexity_pruning_path(X, y)