from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transformation to specified columns after clipping negative values to 0."""

    def __init__(self, columns=None):
        """
        Initialize transformer.

        Args:
            columns (list): Columns to apply log1p transform.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Record feature names and check columns exist."""
        self.feature_names_in_ = X.columns.tolist()

        if self.columns is None:
            raise ValueError(f"Log1pTransformer - columns not provided")
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Log1pTransformer - columns not found: {missing_columns}")
        return self

    def transform(self, X):
        """Clip negative values to 0, then apply log1p transform."""
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].clip(lower=0)
            X_copy[col] = np.log1p(X_copy[col])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return list(getattr(self, "feature_names_in_", []))
