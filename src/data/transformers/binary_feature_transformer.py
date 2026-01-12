from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class BinaryFeature(BaseEstimator, TransformerMixin):
    """Create binary flags based on numeric column values."""

    def __init__(self, columns=None):

        self.columns = columns

    def fit(self, X, y=None):
        """Store original feature names."""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """Add specified numeric columns to binary flags (1 = value > 0, 0 = value == 0)."""
        X_copy = X.copy()
        for col in self.columns:
            split_col = col.split("_")[-1] 
            X_copy[f"has_{split_col}"] = (X_copy[col] > 0).astype('int64') # false->0, true->1
            
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return list(getattr(self, "feature_names_in_", []))
