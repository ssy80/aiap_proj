from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ClipNegative(BaseEstimator, TransformerMixin):
    """Clip negative numeric values to zero."""

    def __init__(self, columns=None):
        """
        Args:
            columns (list): Columns to clip negative values in.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Store feature names."""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """Clip negative values to 0 for specified columns."""
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].clip(lower=0)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return list(getattr(self, "feature_names_in_", []))
