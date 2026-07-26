#!/usr/bin/env python3
"""A function to calculate feature importance."""
import numpy as np

def feature_importance(rf):
    """Arguments:

    rf: A trained Scikit-learn RandomForestClassifier instance.
    Returns:

    importances: A NumPy array of feature importance scores.
    indices: A NumPy array of feature indices sorted from least to most important (ascending order).
    """
    feature_importances = rf.feature_importances_
    indices = np.argsort(feature_importances)
    return feature_importances[indices], indices
