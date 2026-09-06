#!/usr/bin/env python3
"""SVM Classifier with Different Kernels.

This module provides a function to create SVM classifiers with different
kernels (linear, poly, rbf).
"""
from sklearn import svm


def get_SVM_model(name, random_state):
    """Create and return an untrained SVM classifier with the specified kernel.

    Args:
        name: A string indicating the kernel type. Accepted values are:
            'linear': SVM with linear kernel
            'poly': SVM with polynomial kernel
            'rbf': SVM with radial basis function kernel
        random_state: An integer used to set the random seed for
            reproducibility.

    Returns:
        An untrained instance of SVC with the specified kernel.
    """
    if name == 'linear':
        model = svm.SVC(kernel='linear', random_state=random_state)
    elif name == 'poly':
        model = svm.SVC(kernel='poly', random_state=random_state)
    elif name == 'rbf':
        model = svm.SVC(kernel='rbf', random_state=random_state)
    else:
        raise ValueError("name must be one of: 'linear', 'poly', 'rbf'")

    return model
