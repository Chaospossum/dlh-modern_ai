#!/usr/bin/env python3
"""Decision tree visualization module.

This module provides a function to view the decision rules of a trained
tree-based classifier using Scikit-learn.
"""
from sklearn.tree import export_text


def draw(clf, feature_names, class_names):
    """Prints the decision rules of a trained tree classifier.

    Args:
        clf: A trained Scikit-learn tree classifier instance.
        feature_names: List of feature names.
        class_names: List of class names.

    Returns:
        None
    """
    tree_rules = export_text(
        clf,
        feature_names=feature_names,
        class_names=class_names
    )
    print(tree_rules)