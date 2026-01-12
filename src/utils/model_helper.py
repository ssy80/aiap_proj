from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_base_model_class(algorithm: str) -> object:
    """
    Maps an algorithm name to its corresponding model class.

    Raises:
        ValueError: If the algorithm is not supported.
    """

    model_map = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'xgboost': XGBClassifier
    }

    if algorithm not in model_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    base_model_class = model_map[algorithm]

    return base_model_class
