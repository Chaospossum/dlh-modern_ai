#!/usr/bin/env python3
"""Write a function evaluation_metrics_for_regression(y_true, y_pred).

Compute common evaluation metrics for regression tasks using Scikit-learn.
"""
import numpy as np
from sklearn import metrics

def evaluation_metrics_for_regression(y_true, y_pred):
    """Compute common evaluation metrics for regression tasks.

    Args:
        y_true: A 1D NumPy array containing the true target values.
        y_pred: A 1D NumPy array containing the predicted target values.

    Returns:
        A tuple (mse, rmse, mae, r2) where:
        mse: Mean Squared Error.
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        r2: R² Score.
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mse, rmse, mae, r2