import shap

def get_shap_explainer_and_values(model, X_train, X_test):
    """Generate SHAP explainer and SHAP values for a trained regression model.

    Args:
        model: A trained regression model (e.g., Ridge, RandomForestRegressor).
        X_train: Training data (used as background for the explainer).
        X_test: Test data to compute SHAP values for.

    Returns:
        explainer: SHAP explainer object.
        shap_values: SHAP values for X_test.
    """
    # Create the explainer with X_train as background
    explainer = shap.Explainer(model, X_train)

    # Compute SHAP values for X_test
    shap_values = explainer(X_test)

    return explainer, shap_values