#!/usr/bin/env python3
from sklearn import tree

def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """Builds a decision tree classifier with Gini impurity.

    Args:
        min_samples_leaf (int): Minimum number of samples required at a leaf node.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        random_state (int): Seed for reproducibility.

    Returns:
        model: A Scikit-learn DecisionTreeClassifier instance.
    """
    model = tree.DecisionTreeClassifier(
        criterion='gini',
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    return model