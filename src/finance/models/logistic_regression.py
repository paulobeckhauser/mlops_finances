# mypy: disallow-untyped-defs
from sklearn.linear_model import LogisticRegression


def get_logistic_regression_model(random_state: int = 42) -> LogisticRegression:
    """Return a Logistic Regression model."""
    return LogisticRegression(random_state=random_state)
