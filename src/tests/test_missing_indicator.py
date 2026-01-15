import pytest
import pandas as pd
from pandas.testing import assert_series_equal
from data.transformers.missing_indicator_transformer import MissingIndicator


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "a": [1, None, 3, None],
        "b": [10, 20, None, 40],
        "c": [5, 6, 7, 8]
    })


def test_missing_indicator_creates_flags(sample_df):
    """Verify correct missing-value flags are created."""
    transformer = MissingIndicator(columns=["a", "b"])
    result = transformer.fit_transform(sample_df)

    assert result["is_missing_a"].tolist() == [0, 1, 0, 1]
    assert result["is_missing_b"].tolist() == [0, 0, 1, 0]


def test_missing_indicator_handles_non_missing_columns(sample_df):
    """If column has no missing values, indicator should be all zeros."""
    transformer = MissingIndicator(columns=["c"])
    result = transformer.fit_transform(sample_df)

    assert result["is_missing_c"].tolist() == [0, 0, 0, 0]


def test_missing_indicator_adds_new_columns(sample_df):
    """Verify new indicator columns are appended, not replacing original columns."""
    transformer = MissingIndicator(columns=["a"])
    result = transformer.fit_transform(sample_df)

    assert "a" in result.columns
    assert "is_missing_a" in result.columns


def test_missing_indicator_flag_dtype(sample_df):
    """Ensure indicator column is int64."""
    transformer = MissingIndicator(columns=["a"])
    result = transformer.fit_transform(sample_df)

    assert result["is_missing_a"].dtype == "int64"


def test_feature_names_out(sample_df):
    """Ensure feature names passthrough is correct."""
    transformer = MissingIndicator(columns=["a"])
    transformer.fit(sample_df)

    assert transformer.get_feature_names_out() == list(sample_df.columns)
