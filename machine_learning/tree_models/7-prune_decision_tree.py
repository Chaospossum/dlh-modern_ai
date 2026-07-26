#!/usr/bin/env python3
"""Train and evaluate trees with different ccp_alpha values."""

from sklearn import tree
train_tree = __import__('1-train').train_tree
generate_predictions = __import__('3-generate_predictions').generate_predictions


def prune_and_evaluate_trees(
        X_train, y_train, X_test, y_test, ccp_alphas,
        random_state, min_samples_leaf, min_samples_split):
    """Train and evaluate trees with different ccp_alpha values.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        ccp_alphas: Array of ccp_alpha values.
        random_state: Random seed.
        min_samples_leaf: Min samples per leaf.
        min_samples_split: Min samples to split.

    Returns:
        tuple: (clfs, train_scores, test_scores)
    """
    clfs = []          # List to store trained classifiers
    train_scores = []  # List to store training accuracies
    test_scores = []   # List to store testing accuracies

    for alpha in ccp_alphas:
        # 1. Create a new DecisionTreeClassifier with:
        #    - ccp_alpha = alpha
        #    - min_samples_leaf = min_samples_leaf
        #    - min_samples_split = min_samples_split
        #    - random_state = random_state
        clf = tree.DecisionTreeClassifier(
            ccp_alpha=alpha,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        # 2. Train the classifier (use `train_tree` from Task 1)
        train_tree(clf, X_train, y_train)

        # 3. Calculate training accuracy:
        #    - Generate predictions for X_train
        #    - Compare to y_train, compute accuracy
        train_pred = generate_predictions(clf, X_train)
        train_acc = float((train_pred == y_train).mean())
        train_scores.append(train_acc)

        # 4. Calculate testing accuracy:
        #    - Generate predictions for X_test
        #    - Compare to y_test, compute accuracy
        test_pred = generate_predictions(clf, X_test)
        test_acc = float((test_pred == y_test).mean())
        test_scores.append(test_acc)

        # 5. Save the trained classifier to the list
        clfs.append(clf)

    return clfs, train_scores, test_scores
