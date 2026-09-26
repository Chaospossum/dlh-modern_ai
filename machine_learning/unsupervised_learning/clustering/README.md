#!/usr/bin/env python3
"""Clustering validation module.

Validates inputs for K-means clustering and outputs status messages.
"""
import numpy as np


def validate(X, k):
    """Validate clustering inputs.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: int containing the number of clusters.

    Returns:
        str: Status message.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return "invalid X"
    if not isinstance(k, int) or k <= 0:
        return "invalid k"
    if X.shape[1] > 10:
        return "High dimensional X"
    return "Normal"


if __name__ == "__main__":
    X_normal = np.array([[1, 2], [3, 4]])
    print(validate(X_normal, 3))
    X_high = np.random.rand(5, 15)
    print(validate(X_high, 3))
    print(validate("invalid", 3))
    print(validate(X_normal, -1))
