#!/usr/bin/env python3
"""A function to create a random forest classifier."""
from sklearn.ensemble import RandomForestClassifier


def random_forest(n_estimators, random_state):
    """Arguments:

    n_estimators: Number of trees in the forest.
    random_state: Seed used by the random number generator for reproducibility.

    Returns:
    model: A Scikit-learn RandomForestClassifier instance."""

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    return model
