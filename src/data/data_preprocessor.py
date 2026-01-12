import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.helper import setup_logging, safe_get
import logging
from sklearn.model_selection import train_test_split
from utils.helper import df_columns_to_snake
from data.imputers.numeric_imputer import NumericImputer
from data.transformers.missing_indicator_transformer import MissingIndicator
from data.transformers.winsorizer import Winsorizer
from data.transformers.log1p_transformer import Log1pTransformer
from data.transformers.clip_negative_transformer import ClipNegative
from data.transformers.lowercase_transformer import LowercaseTransformer
from data.transformers.binary_feature_transformer import BinaryFeature


class DataPreprocessor:
    """
    Builds and runs the full preprocessing pipeline for the dataset.
    """

    def __init__(self, config: dict):
        """
        Initializes the preprocessor with configuration and logging.
        """
        if config is None:
            raise ValueError("DataPreprocessor __init__: config cannot be None")

        self.config = config
        self.data_preprocessor = None

        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Build the preprocessing pipeline for cleaning, transforming, and encoding features.
        
        Steps included:
        - lowercase categorical text
        - add missing-value feature for line_of_code
        - impute missing numeric values
        - add binary features
        - clip negative numeric values to zero
        - winsorize outliers
        - apply log1p transformation
        - one-hot encode categoricals and scale numeric features

        Returns:
            Pipeline: The full preprocessing pipeline.
        """
        config_column_mappings = safe_get(self.config, 'preprocessing', 'column_mappings', required=True)
        
        industry = safe_get(config_column_mappings, 'categorical', 'industry', required=True)
        hosting_provider = safe_get(config_column_mappings, 'categorical', 'hosting_provider', required=True)
        line_of_code = safe_get(config_column_mappings, 'numerical', 'line_of_code', required=True)
        largest_line_length = safe_get(config_column_mappings, 'numerical', 'largest_line_length', required=True)
        no_of_url_redirect = safe_get(config_column_mappings, 'numerical', 'no_of_url_redirect', required=True)
        no_of_self_redirect = safe_get(config_column_mappings, 'numerical', 'no_of_self_redirect', required=True)
        no_of_popup = safe_get(config_column_mappings, 'numerical', 'no_of_popup', required=True)
        no_of_iframe = safe_get(config_column_mappings, 'numerical', 'no_of_iframe', required=True)
        no_of_image = safe_get(config_column_mappings, 'numerical', 'no_of_image', required=True)
        no_of_self_ref = safe_get(config_column_mappings, 'numerical', 'no_of_self_ref', required=True)
        no_of_external_ref = safe_get(config_column_mappings, 'numerical', 'no_of_external_ref', required=True)
        is_robots = safe_get(config_column_mappings, 'numerical', 'is_robots', required=True)
        is_responsive = safe_get(config_column_mappings, 'numerical', 'is_responsive', required=True)   
        domain_age_months = safe_get(config_column_mappings, 'numerical', 'domain_age_months', required=True)

        impute_strategy = safe_get(self.config, 'preprocessing', 'impute_strategy', required=True)

        negative_clip_features = [no_of_image]
        lowercase_features = [industry, hosting_provider]
        winsorize_features = [line_of_code, largest_line_length, no_of_popup, no_of_image, no_of_self_ref, no_of_external_ref, no_of_iframe]
        log1p_features = winsorize_features
        robust_features = log1p_features
        one_hot_features = [industry, hosting_provider]
        add_binary_features = [no_of_popup, no_of_iframe, no_of_image]

        feature_transformer = ColumnTransformer([
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_features),
            ('robust', RobustScaler(), robust_features)
        ], remainder='passthrough')

        preprocessor = Pipeline([
            ('lowercase', LowercaseTransformer(columns=lowercase_features)),
            ('add_missing_loc_feature', MissingIndicator(columns=[line_of_code])),
            ('impute_loc', NumericImputer(
                columns=[line_of_code],
                strategy=impute_strategy
            )),
            ("add_binary_feature", BinaryFeature(columns=add_binary_features)),
            ('clip_negative', ClipNegative(columns=negative_clip_features)),
            ('winsorize', Winsorizer(columns=winsorize_features, lower=0.01, upper=0.99)),
            ('log1p', Log1pTransformer(columns=log1p_features)),
            ('encode_features', feature_transformer)
        ])

        self.preprocessor = preprocessor

    def preprocess_target(self, y: pd.Series) -> pd.Series:
        """
        Converts the target column to int if needed.
        """
        if y.dtype == "float64":
            y = y.astype(int)
        return y

    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """
        Runs the full preprocessing workflow:
        - Drops identifier column and duplicates
        - Splits into features and target
        - Converts target type
        - Builds preprocessing pipeline
        - Splits into train/test
        - Fits pipeline on train and transforms both
        Returns processed train/test sets and the pipeline.
        """
        df = self.pre_split_processing(df)

        label = safe_get(self.config, 'preprocessing', 'column_mappings', 'target', 'label', required=True)

        X = df.drop(columns=[label])
        y = df[label]

        y = self.preprocess_target(y)

        self.create_preprocessing_pipeline()

        test_size = safe_get(self.config, 'preprocessing', 'test_size', required=True)
        random_state = safe_get(self.config, 'preprocessing', 'random_state', required=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        X_train_processed = pd.DataFrame(
            X_train_processed,
            columns=self.preprocessor.get_feature_names_out(),
            index=X_train.index
        )

        X_test_processed = pd.DataFrame(
            X_test_processed,
            columns=self.preprocessor.get_feature_names_out(),
            index=X_test.index
        )

        self.logger.info(
            f"Data preprocessing completed. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}"
        )

        return X_train_processed, X_test_processed, y_train, y_test, self.preprocessor

    def pre_split_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the entire dataset before splitting.
        Returns the processed dataframe
        """
        unnamed_id = safe_get(self.config, 'preprocessing', 'column_mappings', 'identifier', 'unnamed', required=True)

        # Drop identifier column and duplicated rows
        df = df.drop(columns=[unnamed_id])
        df = df.drop_duplicates()
        assert df.duplicated().sum() == 0, "DataPreprocessor pre_split_processing: Duplicated rows still exist after dropping duplicates"

        # Column name to snake case
        df = df_columns_to_snake(df)

        # Rename column name - "no_ofi_frame" to ""no_of_iframe", "robots" to "is_robots"
        df = df.rename(columns={"no_ofi_frame": "no_of_iframe", "robots": "is_robots"})
        
        return df
