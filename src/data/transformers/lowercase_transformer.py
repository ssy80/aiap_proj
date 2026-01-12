from sklearn.base import BaseEstimator, TransformerMixin

class LowercaseTransformer(BaseEstimator, TransformerMixin):
    """Convert string values in specified columns to lowercase."""

    def __init__(self, columns=None):
        """
        Initialize transformer.

        Args:
            columns (list): Columns to convert to lowercase.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Record feature names and check columns exist and are of string type."""
        self.feature_names_in_ = X.columns.tolist()

        if self.columns is None:
            raise ValueError("LowercaseTransformer - columns not provided")
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"LowercaseTransformer - columns not found: {missing_columns}")
        invalid_columns = [col for col in self.columns if X[col].dtype != 'object']
        if invalid_columns:
            raise ValueError(f"LowercaseTransformer - invalid column dtype: {invalid_columns}")
        return self

    def transform(self, X):
        """Convert specified columns to lowercase."""
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].str.strip().str.lower()
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        return list(input_features)
