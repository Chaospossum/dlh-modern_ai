#!/usr/bin/env python3
"""Logistic Regression Model Module.

This module provides a function to create an untrained LogisticRegression model
for binary classification tasks using Scikit-learn.
"""
from sklearn import linear_model


def logistic_regression(random_state):
    """Create and return an untrained LogisticRegression model.

    Logistic Regression is a linear model for binary classification that fits
    a logistic function to model the probability that a given input belongs
    to a particular class.

    Args:
        random_state: An integer used to set the random seed for
            reproducibility in the model's optimization process.

    Returns:
        An untrained LogisticRegression instance. The model supports:
        - get_params() for hyperparameters
        - Checking fitted state via hasattr(model, 'coef_')
        - predict() for class labels
        - predict_proba() for class probabilities
    """
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
