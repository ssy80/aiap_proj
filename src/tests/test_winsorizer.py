import pytest
import pandas as pd
import numpy as np
from data.transformers.winsorizer import Winsorizer


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing winsorization."""
    return pd.DataFrame({
        "a": [1, 2, 3, 100, 200],
        "b": [10, 20, 30, 40, 1000],
        "other": [5, 6, 7, 8, 9]
    })


def test_winsorizer_computes_bounds(sample_df):
    """Test that fit() computes correct percentile bounds."""
    transformer = Winsorizer(columns=["a", "b"], lower=0.01, upper=0.99)
    transformer.fit(sample_df)

    # manually compute expected quantiles
    expected_a = (
        sample_df["a"].quantile(0.01),
        sample_df["a"].quantile(0.99)
    )
    expected_b = (
        sample_df["b"].quantile(0.01),
        sample_df["b"].quantile(0.99)
    )

    assert transformer.bounds_["a"] == expected_a
    assert transformer.bounds_["b"] == expected_b


def test_winsorizer_clips_values(sample_df):
    """Ensure transform() clips values to the learned bounds."""
    transformer = Winsorizer(columns=["a"], lower=0.01, upper=0.99)
    transformer.fit(sample_df)

    result = transformer.transform(sample_df)
    low, high = transformer.bounds_["a"]

    # Check no value is outside the learned bounds
    assert (result["a"] >= low).all()
    assert (result["a"] <= high).all()
