from .deep_learning import FeedforwardNN
from .logistic_regression import get_logistic_regression_model
from .random_forest import get_random_forest_model


def get_model(model_name, **kwargs):
    """
    Factory method to get a model by name.
    Args:
        model_name (str): The name of the model.
        kwargs: Additional arguments for model initialization.
    Returns:
        model: The initialized model.
    """
    if model_name == "random_forest":
        return get_random_forest_model(**kwargs)
    elif model_name == "logistic_regression":
        return get_logistic_regression_model(**kwargs)
    elif model_name == "deep_learning":
        input_size = kwargs.get("input_size", 10)  # Default to 10 features if not provided
        hidden_size = kwargs.get("hidden_size", 64)
        output_size = kwargs.get("output_size", 2)
        return FeedforwardNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
