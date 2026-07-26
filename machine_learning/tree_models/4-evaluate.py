#!/usr/bin/env python3
"""
One-line summary of what this script does.

Longer description if you need it: what it takes in, what it spits out.
"""

from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """Generate classification report for the given true and predicted labels.

    Args:
        true_labels: Array-like of true labels.
        predicted_labels: Array-like of predicted labels.
        class_names: List of class names (e.g., ['class_0', 'class_1']).

    Returns:
        str: Classification report as a string.
    """
    report = metrics.classification_report(
        true_labels, predicted_labels, target_names=class_names,
        output_dict=False
    )
    return report
