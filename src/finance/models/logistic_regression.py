from sklearn.linear_model import LogisticRegression


def get_logistic_regression_model(random_state=42):
    """Return a Logistic Regression model."""
    return LogisticRegression(random_state=random_state)