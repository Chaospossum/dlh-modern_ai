#!/usr/bin/env python3
"""Initialize cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    return np.random.uniform(low, high, size=(k, X.shape[1]))
