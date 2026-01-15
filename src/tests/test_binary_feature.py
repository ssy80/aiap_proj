import pytest
import pandas as pd
from data.transformers.binary_feature_transformer import BinaryFeature


@pytest.fixture
def sample_df():
    """Sample DataFrame containing numeric values."""
    return pd.DataFrame({
        "count_a": [0, 1, 5, -3, 0],
        "metric_b": [10, 0, -1, 2, 0],
        "other": [100, 200, 300, 400, 500]
    })


def test_binary_feature_creates_correct_flags(sample_df):
    """Ensure binary flags are created with correct logic and names."""
    transformer = BinaryFeature(columns=["count_a", "metric_b"])
    result = transformer.fit_transform(sample_df)

    assert result["has_a"].tolist() == [0, 1, 1, 0, 0]
    assert result["has_b"].tolist() == [1, 0, 0, 1, 0]


def test_binary_feature_creates_correct_column_names(sample_df):
    """Ensure new column names match 'has_<suffix>'."""
    transformer = BinaryFeature(columns=["count_a"])
    result = transformer.fit_transform(sample_df)

    assert "has_a" in result.columns
    assert "has_count" not in result.columns   # Should not use full prefix


def test_binary_feature_non_target_columns_unchanged(sample_df):
    """Ensure non-target columns are untouched."""
    transformer = BinaryFeature(columns=["count_a"])
    result = transformer.fit_transform(sample_df)

    assert result["other"].equals(sample_df["other"])


def test_binary_feature_dtype_is_int64(sample_df):
    """Binary flag dtype should be int64."""
    transformer = BinaryFeature(columns=["count_a"])
    result = transformer.fit_transform(sample_df)

    assert result["has_a"].dtype == "int64"


def test_binary_feature_multiple_columns(sample_df):
    """Ensure transformer correctly handles multiple input columns."""
    transformer = BinaryFeature(columns=["count_a", "metric_b"])
    result = transformer.fit_transform(sample_df)

    assert "has_a" in result.columns
    assert "has_b" in result.columns


def test_feature_names_out(sample_df):
    """Feature name passthrough should return original columns."""
    transformer = BinaryFeature(columns=["count_a"])
    transformer.fit(sample_df)

    assert transformer.get_feature_names_out() == list(sample_df.columns)