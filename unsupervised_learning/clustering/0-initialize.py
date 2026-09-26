#!/usr/bin/env python3
"""Initialize cluster centroids for K-means clustering.

This module provides a function to initialize centroids for the K-means
clustering algorithm using a multivariate uniform distribution.
"""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means clustering.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number
            of data points and d is the number of dimensions.
        k (int): Number of clusters (positive integer).

    Returns:
        numpy.ndarray or None: Array of shape (k, d) containing initialized
            centroids, or None if inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    return np.random.uniform(low, high, size=(k, X.shape[1]))
