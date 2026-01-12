import joblib
import os
import logging
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from utils.helper import setup_logging, safe_get
from utils.model_helper import get_base_model_class


class ModelTrainer:
    """Handles model training, tuning, and saving."""

    def __init__(self, config: dict):
        """
        Initialize ModelTrainer with configuration.

        Args:
            config (dict): Configuration settings for training and tuning.
        """
        if config is None:
            raise ValueError("ModelTrainer __init__: config cannot be None")

        self.config = config
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def train_model(self, X_train, y_train, best_params=None):
        """
        Train the model, optionally using best hyperparameters and cross-validation.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            best_params (dict, optional): Best hyperparameters from tuning.

        Returns:
            tuple: (trained_model, cv_scores)
        """
        cross_validate = safe_get(self.config, 'training', 'cross_validate', required=True)

        # Initialize model
        algorithm = safe_get(self.config, 'model', 'algorithm', required=True)
        base_model_class = get_base_model_class(algorithm)
        if best_params:
            self.model = base_model_class(**best_params)
        else:
            hyperparams = safe_get(self.config, 'model', 'hyperparameters', algorithm, required=True)
            self.model = base_model_class(**hyperparams)

        self.logger.info(f"Model for training: {self.model}")
        self.logger.info(f"Cross-validation: {cross_validate}")

        # Optional cross-validation
        cv_scores = None
        if cross_validate:
            cv = safe_get(self.config, 'training', 'cv_folds', required=True)
            scoring = safe_get(self.config, 'training', 'scoring_metric', required=True)
            self.logger.info(f"Running cross-validation: cv={cv}, scoring={scoring}")

            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
            self.logger.info(f"CV scores ({scoring}): {cv_scores}")
            self.logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train model
        self.logger.info("Training model...")
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")

        return self.model, cv_scores

    def find_best_parameters(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            dict: Best found hyperparameters.
        """
        self.logger.info("Starting hyperparameter tuning")

        algorithm = safe_get(self.config, 'model', 'algorithm', required=True)
        base_model_class = get_base_model_class(algorithm)
        param_grid = safe_get(self.config, 'tuning', 'hyperparameters', algorithm, required=True)
        search_strategy = safe_get(self.config, 'tuning', 'search_strategy', required=True)

        model = base_model_class()
        self.logger.info(f"Using {search_strategy} search for {algorithm}")

        if search_strategy == 'random':
            best_params = self._randomized_search(X_train, y_train, model, param_grid)
        elif search_strategy == 'grid':
            best_params = self._grid_search(X_train, y_train, model, param_grid)
        else:
            raise ValueError("Invalid search strategy in config")

        self.logger.info(f"Best parameters found: {best_params}")
        return best_params

    def _randomized_search(self, X_train, y_train, base_model, param_distributions):
        """
        Run RandomizedSearchCV for faster hyperparameter tuning.

        Returns:
            dict: Best parameters from search.
        """
        self.logger.info("Running RandomizedSearchCV")

        scoring = safe_get(self.config, 'training', 'scoring_metric', required=True)
        cv = safe_get(self.config, 'training', 'cv_folds', required=True)
        n_iter = safe_get(self.config, 'tuning', 'n_iter', required=True)
        n_jobs = safe_get(self.config, 'tuning', 'n_jobs', required=True)
        random_state = safe_get(self.config, 'tuning', 'random_state', required=True)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            return_train_score=True
        )
        search.fit(X_train, y_train)
        return search.best_params_

    def _grid_search(self, X_train, y_train, base_model, param_grid):
        """
        Run exhaustive GridSearchCV to find best parameters.

        Returns:
            dict: Best parameters from grid search.
        """
        self.logger.info("Running GridSearchCV")

        scoring = safe_get(self.config, 'training', 'scoring_metric', required=True)
        cv = safe_get(self.config, 'training', 'cv_folds', required=True)
        n_jobs = safe_get(self.config, 'tuning', 'n_jobs', required=True)

        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True
        )
        search.fit(X_train, y_train)
        return search.best_params_

    def save_model(self, preprocessor=None, features_to_drop=None, selector=None, model=None):
        """
        Save the trained model and preprocessing components.

        Args:
            preprocessor: Preprocessing pipeline used.
            features_to_drop (list): Features dropped before training.
            selector: Feature selector object.
            model: Trained model to save.
        """
        save_model = safe_get(self.config, 'training', 'save_model', required=True)
        output_path = safe_get(self.config, 'training', 'model_output_path', required=True)

        if save_model:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pipeline_obj = {
                'model': model,
                'preprocessor': preprocessor,
                'features_to_drop': features_to_drop,
                'selector': selector
            }
            joblib.dump(pipeline_obj, output_path)
            self.logger.info(f"Model saved to {output_path}")
