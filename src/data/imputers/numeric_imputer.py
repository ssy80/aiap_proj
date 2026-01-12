from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class NumericImputer(BaseEstimator, TransformerMixin):
    """Custom numeric imputer supporting mean or median strategy."""

    def __init__(self, columns=None, strategy='median'):
        """
        Args:
            columns (list): List of numeric columns to impute.
            strategy (str): 'mean' or 'median'.
        """
        self.columns = columns
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, X, y=None):
        """Learn median/mean from training data only."""
        self.feature_names_in_ = X.columns.tolist()
        X_copy = X.copy()

        for col in self.columns:
            if self.strategy == 'median':
                self.fill_values_[col] = X_copy[col].median()
            elif self.strategy == 'mean':
                self.fill_values_[col] = X_copy[col].mean()
            else:
                raise ValueError("strategy must be 'mean' or 'median'")

        return self

    def transform(self, X):
        """Apply imputation to missing numeric values."""
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].fillna(self.fill_values_[col])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        return list(getattr(self, "feature_names_in_", []))
