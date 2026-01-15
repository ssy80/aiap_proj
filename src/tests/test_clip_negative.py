import pytest
import pandas as pd
import numpy as np
from data.transformers.clip_negative_transformer import ClipNegative

@pytest.fixture
def sample_df():
    """Fixture: example dataframe containing negatives and positives."""
    return pd.DataFrame({
        "feature_a": [-5, 0, 10, -3],
        "feature_b": [2.5, -1.5, 3.0, -0.2],
        "not_clipped": [1, 2, 3, 4]
    })


def test_clipnegative_clips_values(sample_df):
    """Verify negative values are clipped to zero."""
    transformer = ClipNegative(columns=["feature_a", "feature_b"])
    result = transformer.fit_transform(sample_df)

    assert result["feature_a"].tolist() == [0, 0, 10, 0]
    assert result["feature_b"].tolist() == [2.5, 0, 3.0, 0]


def test_clipnegative_does_not_modify_other_columns(sample_df):
    """Ensure non-target columns remain unchanged."""
    transformer = ClipNegative(columns=["feature_a"])
    result = transformer.fit_transform(sample_df)

    assert result["not_clipped"].equals(sample_df["not_clipped"])


def test_clipnegative_handles_float_and_int(sample_df):
    """Ensure clipping works for both int and float dtype columns."""
    transformer = ClipNegative(columns=["feature_a", "feature_b"])
    result = transformer.fit_transform(sample_df)

    # all values must be >= 0
    assert (result["feature_a"] >= 0).all()
    assert (result["feature_b"] >= 0).all()


def test_feature_names_out(sample_df):
    """Check feature name passthrough logic."""
    transformer = ClipNegative(columns=["feature_a"])
    transformer.fit(sample_df)

    assert transformer.get_feature_names_out() == list(sample_df.columns)
