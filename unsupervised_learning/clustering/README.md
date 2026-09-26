#!/usr/bin/env python3
"""Test script for clustering initialization."""
import numpy as np
initialize = __import__('0-initialize').initialize

if __name__ == "__main__":
    np.random.seed(0)
    # Test with valid low-dimensional input
    X_normal = np.array([[1, 2], [3, 4], [5, 6]])
    result = initialize(X_normal, 3)
    if result is not None and result.shape == (3, 2):
        print("Normal")

    # Test with valid high-dimensional input
    X_high = np.random.rand(10, 15)
    result = initialize(X_high, 3)
    if result is not None and result.shape == (3, 15):
        print("High dimensional X")

    # Test with invalid X
    result = initialize("invalid", 3)
    if result is None:
        print("invalid X")

    # Test with invalid k
    result = initialize(X_normal, -1)
    if result is None:
        print("invalid k")
