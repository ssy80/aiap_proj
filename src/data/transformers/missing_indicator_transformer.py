from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class MissingIndicator(BaseEstimator, TransformerMixin):
    """Create binary flags indicating missing values."""

    def __init__(self, columns=None):
        """
        Args:
            columns (list): Columns to create missing-value indicators for.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Store original feature names."""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """Add new indicator columns (1 = missing, 0 = not missing)."""
        X_copy = X.copy()
        for col in self.columns:
            X_copy[f"is_missing_{col}"] = X_copy[col].isna().astype('int64') # false->0, true->1
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return list(getattr(self, "feature_names_in_", []))
