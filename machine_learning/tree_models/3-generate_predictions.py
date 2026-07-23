#!/usr/bin/env python3
"""Generate predictions module.

This module provides a function to generate predictions from a trained
tree-based classifier using Scikit-learn.
"""


def generate_predictions(clf, X):
    """Generates predictions from a trained tree-based classifier.

    Args:
        clf: A trained Scikit-learn classifier instance.
        X: Input features.

    Returns:
        predictions: Array of predictions.
    """
    return clf.predict(X)
