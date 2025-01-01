from sklearn.ensemble import RandomForestClassifier

def get_random_forest_model(random_state=42):
    """Return a Random Forest model."""
    return RandomForestClassifier(random_state=random_state)