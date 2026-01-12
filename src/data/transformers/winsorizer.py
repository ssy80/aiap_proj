from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns by capping extreme values at given percentiles."""

    def __init__(self, columns=None, lower=0.01, upper=0.99):
        """
        Args:
            columns (list): Columns to winsorize.
            lower (float): Lower percentile (0-1).
            upper (float): Upper percentile (0-1).
        """
        self.columns = columns
        self.lower = lower
        self.upper = upper
        self.bounds_ = {}

    def fit(self, X, y=None):
        """Compute lower and upper bounds per column."""
        self.feature_names_in_ = X.columns.tolist()
        X_copy = X.copy()

        for col in self.columns:
            low_val = X_copy[col].quantile(self.lower)
            high_val = X_copy[col].quantile(self.upper)
            self.bounds_[col] = (low_val, high_val)

        return self

    def transform(self, X):
        """Cap values outside the learned bounds."""
        X_copy = X.copy()

        for col in self.columns:
            low, high = self.bounds_[col]
            X_copy[col] = np.clip(X_copy[col], low, high)

        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return list(getattr(self, "feature_names_in_", []))
