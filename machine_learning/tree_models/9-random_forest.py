#!/usr/bin/env python3
"""A function to create a random forest classifier."""
from pyexpat import model

from sklearn import ensemble

def random_forest(n_estimators, random_state):
    """Arguments:

    n_estimators: Number of trees in the forest.
    random_state: Seed used by the random number generator for reproducibility.

    Returns:
    model: A Scikit-learn RandomForestClassifier instance."""

    model = ensemble.RandomForestClassifier(n_estimators=20, random_state=40000)
return model