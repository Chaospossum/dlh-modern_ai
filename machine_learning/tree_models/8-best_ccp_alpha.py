#!/usr/bin/env python3
"""Select the best ccp_alpha for post-pruning a decision tree."""

import numpy as np
from sklearn import ensemble


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Select the best ccp_alpha based on test accuracy and generalization.

    Args:
        clfs: List of trained DecisionTreeClassifier instances.
        train_scores: List of training accuracy scores.
        test_scores: List of test accuracy scores.
        ccp_alphas: List of ccp_alpha values.

    Returns:
        tuple: (best_alpha, best_clf)
    """
    # Step 1: Find the index of the best model
    best_index = max(
        range(len(clfs)),
        key=lambda i: (
            test_scores[i],  # 1. Maximize test accuracy
            -(train_scores[i] - test_scores[i]),  # 2. Minimize train-test gap
            ccp_alphas[i]  # 3. Maximize ccp_alpha
        )
    )

    # Step 2: Return the best alpha and classifier
    return ccp_alphas[best_index], clfs[best_index]
