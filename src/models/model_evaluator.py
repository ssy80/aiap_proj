import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import os
from utils.helper import setup_logging, safe_get
import logging


class ModelEvaluator:
    """Evaluates a trained model and saves performance reports."""

    def __init__(self, config: dict):
        """
        Initialize the evaluator with configuration settings.

        Args:
            config (dict): Configuration dictionary containing evaluation options.
        """
        if config is None:
            raise ValueError("ModelEvaluator __init__: config cannot be None")

        self.config = config
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a trained model on the test set.

        Calculates metrics like accuracy, precision, recall, F1, ROC-AUC, and
        generates a confusion matrix and classification report.

        Args:
            model: Trained model to evaluate.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True test labels.

        Returns:
            tuple: (metrics_dict, confusion_matrix, classification_report)
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        # Compute metrics
        metrics = {}
        evaluation_metrics = safe_get(self.config, 'evaluation', 'metrics', required=True)

        if 'accuracy' in evaluation_metrics:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        if 'precision' in evaluation_metrics:
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        if 'recall' in evaluation_metrics:
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        if 'f1' in evaluation_metrics:
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
        if 'f1_macro' in evaluation_metrics:
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        if 'roc_auc' in evaluation_metrics and y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1], multi_class='ovr')

        # Generate reports
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        self.logger.info("Model evaluation completed")

        # Save reports if configured
        save_reports = safe_get(self.config, 'evaluation', 'save_reports', required=True)
        if save_reports:
            self.save_evaluation_reports(metrics, cm, class_report, y_test, y_pred)

        return metrics, cm, class_report

    def save_evaluation_reports(self, metrics, cm, class_report, y_test, y_pred):
        """
        Save metrics, classification report, and confusion matrix plot to files.

        Args:
            metrics (dict): Computed evaluation metrics.
            cm (np.ndarray): Confusion matrix.
            class_report (dict): Classification report dictionary.
            y_test (pd.Series): True test labels.
            y_pred (pd.Series): Predicted labels.
        """
        reports_path = safe_get(self.config, 'evaluation', 'reports_path', required=True)
        os.makedirs(reports_path, exist_ok=True)

        # Save metrics
        pd.DataFrame([metrics]).to_csv(f"{reports_path}/metrics.csv", index=False)

        # Save classification report
        pd.DataFrame(class_report).transpose().to_csv(f"{reports_path}/classification_report.csv")

        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{reports_path}/confusion_matrix.png")
        plt.close()

        self.logger.info(f"Evaluation reports saved to {reports_path}")
