#!/usr/bin/env python3
"""a function generate_predictions(clf, X) to generate predictions from a trained tree-based classifier using Scikit-learn"""

def generate_predictions(clf, X):
    """Generates predictions from a trained tree-based classifier."""
    return clf.predict(X)