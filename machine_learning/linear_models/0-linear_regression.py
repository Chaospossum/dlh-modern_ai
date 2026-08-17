#!/usr/bin/env python3
"""Linear regression model builder module.

This module provides a function to create a linear regression model
using scikit-learn.
"""
from sklearn import linear_model


def Linear_Regression():
    """Creates a linear regression model.

    Returns:
        model: An untrained LinearRegression instance.
    """
    model = linear_model.LinearRegression()
    return model
