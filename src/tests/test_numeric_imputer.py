import pytest
import pandas as pd
import numpy as np
from data.imputers.numeric_imputer import NumericImputer


@pytest.fixture
def sample_df():
    """Sample DataFrame containing numeric values with missing entries."""
    return pd.DataFrame({
        "a": [1.0, None, 3.0, None],
        "b": [10, 20, None, 40],
        "c": [100, 200, 300, 400]
    })


def test_numeric_imputer_median(sample_df):
    """Check that median is computed and used for imputation."""
    transformer = NumericImputer(columns=["a", "b"], strategy="median")
    result = transformer.fit_transform(sample_df)

    expected_a = sample_df["a"].median()
    expected_b = sample_df["b"].median()

    assert result["a"].tolist() == [1.0, expected_a, 3.0, expected_a]
    assert result["b"].tolist() == [10, 20, expected_b, 40]


def test_numeric_imputer_mean(sample_df):
    """Check that mean is computed and used for imputation."""
    transformer = NumericImputer(columns=["a"], strategy="mean")
    result = transformer.fit_transform(sample_df)

    expected = sample_df["a"].mean()
    assert result["a"].tolist() == [1.0, expected, 3.0, expected]


def test_numeric_imputer_non_target_columns_unchanged(sample_df):
    """Ensure non-target columns are not modified."""
    transformer = NumericImputer(columns=["a"], strategy="median")
    result = transformer.fit_transform(sample_df)

    assert result["c"].equals(sample_df["c"])


def test_numeric_imputer_multiple_columns(sample_df):
    """Verify imputer works with multiple columns."""
    transformer = NumericImputer(columns=["a", "b"], strategy="mean")
    result = transformer.fit_transform(sample_df)

    expected_a = sample_df["a"].mean()
    expected_b = sample_df["b"].mean()

    assert result["a"].tolist() == [1.0, expected_a, 3.0, expected_a]
    assert result["b"].tolist() == [10, 20, expected_b, 40]


def test_numeric_imputer_invalid_strategy(sample_df):
    """Ensure invalid strategy raises a ValueError."""
    transformer = NumericImputer(columns=["a"], strategy="invalid")

    with pytest.raises(ValueError) as exc:
        transformer.fit(sample_df)

    assert "strategy must be 'mean' or 'median'" in str(exc.value)


def test_feature_names_out(sample_df):
    """Ensure feature name passthrough is correct."""
    transformer = NumericImputer(columns=["a"], strategy="median")
    transformer.fit(sample_df)

    assert transformer.get_feature_names_out() == list(sample_df.columns)
