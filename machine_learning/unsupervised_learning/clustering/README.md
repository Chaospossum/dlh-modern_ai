#!/usr/bin/env python3
"""Clustering initialization validation script.

This script validates input parameters for K-means clustering initialization
and prints appropriate status messages based on the input validity.
"""
import numpy as np


def validate_inputs(X, k):
    """Validate clustering inputs and return status message.

    Args:
        X: Input data array to validate.
        k: Number of clusters to validate.

    Returns:
        str: Status message indicating input validity.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return "invalid X"
    if not isinstance(k, int) or k <= 0:
        return "invalid k"
    if X.shape[1] > 10:
        return "High dimensional X"
    return "Normal"


if __name__ == "__main__":
    # Test case 1: Valid low-dimensional input
    valid_X = np.array([[1, 2], [3, 4], [5, 6]])
    valid_k = 3
    print(validate_inputs(valid_X, valid_k))

    # Test case 2: Valid high-dimensional input
    high_dim_X = np.random.rand(10, 15)
    print(validate_inputs(high_dim_X, valid_k))

    # Test case 3: Invalid X (not 2D array)
    invalid_X = "not an array"
    print(validate_inputs(invalid_X, valid_k))

    # Test case 4: Invalid k
    invalid_k = -1
    print(validate_inputs(valid_X, invalid_k))
