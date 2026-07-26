#!/usr/bin/env python3
"""A function to initialize boosting classifiers."""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """Initialize and return an untrained boosting classifier.

    Arguments:

    name (str): Name of the boosting algorithm. Must be one of:
        'adaboost' — returns an AdaBoostClassifier
        'gradientboosting' — returns a GradientBoostingClassifier
        'xgboost' — returns an XGBClassifier
        'lightgbm' — returns an LGBMClassifier

    n_estimators (int): Number of boosting iterations (trees).

    random_state (int): Random seed for reproducibility.

    Returns:

    An untrained instance of the selected boosting classifier.

    Raises:

    ValueError: If the provided model name is invalid.
    """
    if name == 'adaboost':
        return ensemble.AdaBoostClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == 'gradientboosting':
        return ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == 'xgboost':
        return xgb.XGBClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == 'lightgbm':
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name '{name}'")
