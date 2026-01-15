import pytest
import pandas as pd
from data.transformers.lowercase_transformer import LowercaseTransformer


@pytest.fixture
def sample_df():
    """Fixture: sample dataframe similar to dataset."""
    return pd.DataFrame({
        "industry": ["Banking", " eCommerce ", "manufacturing", None],
        "hosting_provider": ["AWS", "Google", " azure ", "IBM"],
        "numeric_feature": [1.2, 3.4, 5.6, 7.8]
    })


def test_lowercase_transformer_basic(sample_df):
    """Test that specified columns are properly lowercased."""
    transformer = LowercaseTransformer(columns=["industry", "hosting_provider"])
    result = transformer.fit_transform(sample_df)

    assert result["industry"].tolist() == ["banking", "ecommerce", "manufacturing", None]
    assert result["hosting_provider"].tolist() == ["aws", "google", "azure", "ibm"]


def test_lowercase_transformer_ignores_non_string(sample_df):
    """Ensure numeric columns remain unchanged."""
    transformer = LowercaseTransformer(columns=["industry", "hosting_provider"])
    result = transformer.fit_transform(sample_df)

    assert result["numeric_feature"].equals(sample_df["numeric_feature"])


def test_lowercase_transformer_raises_if_column_missing(sample_df):
    """Transformer should error if a column does not exist."""
    transformer = LowercaseTransformer(columns=["industry", "does_not_exist"])

    with pytest.raises(ValueError) as exc:
        transformer.fit(sample_df)

    assert "columns not found" in str(exc.value)


def test_lowercase_transformer_raises_if_non_string_dtype(sample_df):
    """
    Transformer should error if column is not object dtype.
    numeric_feature is float → should fail.
    """
    transformer = LowercaseTransformer(columns=["numeric_feature"])

    with pytest.raises(ValueError) as exc:
        transformer.fit(sample_df)

    assert "invalid column dtype" in str(exc.value)


def test_feature_names_out(sample_df):
    """Check feature name passthrough."""
    transformer = LowercaseTransformer(columns=["industry"])
    transformer.fit(sample_df)

    assert transformer.get_feature_names_out() == list(sample_df.columns)
