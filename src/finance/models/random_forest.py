# mypy: disallow-untyped-defs
from sklearn.ensemble import RandomForestClassifier


def get_random_forest_model(random_state: int = 42) -> RandomForestClassifier:
    """Return a Random Forest model."""
    return RandomForestClassifier(random_state=random_state)
