import pytest
import pandas as pd
import numpy as np
from data.transformers.log1p_transformer import Log1pTransformer


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "a": [-5, 0, 10, 3],
        "b": [1.5, -2.0, 5.0, 0.0],
        "other": [100, 200, 300, 400]
    })


def test_log1p_transformer_applies_log1p(sample_df):
    """Ensure log1p is correctly applied to non-negative values."""
    transformer = Log1pTransformer(columns=["a", "b"])
    result = transformer.fit_transform(sample_df)

    expected_a = np.log1p(np.clip(sample_df["a"], 0, None))
    expected_b = np.log1p(np.clip(sample_df["b"], 0, None))

    assert np.allclose(result["a"], expected_a, equal_nan=True)
    assert np.allclose(result["b"], expected_b, equal_nan=True)


def test_log1p_transformer_clips_negatives(sample_df):
    """Negative values must be clipped to zero before log1p."""
    transformer = Log1pTransformer(columns=["a"])
    result = transformer.fit_transform(sample_df)

    assert result["a"].tolist()[0] == np.log1p(0)


def test_log1p_transformer_does_not_modify_other_columns(sample_df):
    """Non-target columns should remain unchanged."""
    transformer = Log1pTransformer(columns=["a"])
    result = transformer.fit_transform(sample_df)

    assert result["other"].equals(sample_df["other"])


def test_log1p_transformer_raises_if_missing_column(sample_df):
    """Transformer should raise an error if provided column does not exist."""
    transformer = Log1pTransformer(columns=["missing_col"])

    with pytest.raises(ValueError) as exc:
        transformer.fit(sample_df)

    assert "columns not found" in str(exc.value)


def test_feature_names_out(sample_df):
    """Ensure feature name passthrough is correct."""
    transformer = Log1pTransformer(columns=["a"])
    transformer.fit(sample_df)

    assert transformer.get_feature_names_out() == list(sample_df.columns)
