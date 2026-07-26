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

    model = RandomForestClassifier(n_estimators=200, random_state=0)
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 200, 'n_jobs': None, 'oob_score': False, 'random_state': 0, 'verbose': 0, 'warm_start': False}
return model