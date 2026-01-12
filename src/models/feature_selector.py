from sklearn.feature_selection import SelectFromModel
import pandas as pd
from utils.helper import safe_get
from utils.model_helper import get_base_model_class


class FeatureSelector:
    """Selects important features using a model-based method."""

    def __init__(self, config: dict):
        """
        Initialize the feature selector.

        Args:
            config (dict): Configuration containing model details and selection threshold.
        """
        if config is None:
            raise ValueError("FeatureSelector __init__: config cannot be None")

        self.config = config
        self.threshold = safe_get(self.config, 'feature_selection', 'feature_selection_threshold', required=True)
        self.selector = None
        self.selected_columns = None

        if self.threshold is None:
            raise ValueError("FeatureSelector __init__: threshold cannot be None")

    def select_features(self, X_train, y_train, X_test):
        """
        Fit a model to training data and keep only the most important features.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Test features.

        Returns:
            tuple: (X_train_selected, X_test_selected, fitted_selector)
        """
        model = self.get_model()

        # Initialize and fit selector
        self.selector = SelectFromModel(model, threshold=self.threshold)
        self.selector.fit(X_train, y_train)

        # Apply feature selection
        X_train_sel = self.selector.transform(X_train)
        X_test_sel = self.selector.transform(X_test)

        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_columns = X_train.columns[mask].tolist()

        # Convert back to DataFrames
        X_train_sel = pd.DataFrame(X_train_sel, columns=self.selected_columns, index=X_train.index)
        X_test_sel = pd.DataFrame(X_test_sel, columns=self.selected_columns, index=X_test.index)

        return X_train_sel, X_test_sel, self.selector

    def get_model(self):
        """
        Create and return the model used for feature importance.

        Returns:
            sklearn model: The model instance defined in config.
        """
        algorithm = safe_get(self.config, 'model', 'algorithm', required=True)
        hyperparams = safe_get(self.config, 'model', 'hyperparameters', algorithm, required=True)

        base_model_class = get_base_model_class(algorithm)
        model = base_model_class(**hyperparams)
        return model
