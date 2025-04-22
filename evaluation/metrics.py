import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve, confusion_matrix,
                            classification_report)
from logging import getLogger, INFO, Formatter, StreamHandler
import os
import pandas as pd
# Configure logger
eval_logger = getLogger("PerformanceAnalyzer")
eval_logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
eval_logger.addHandler(handler)

class PerformanceAnalyzer:
    """Analyzes the performance of classification models and generates visualizations."""

    def __init__(self, model_dict):
        """
        Initialize with a dictionary of trained model pipelines.

        Args:
            model_dict (dict): Dictionary of {model_name: pipeline}.
        """
        self.model_dict = model_dict
        self.performance_metrics = {}
        eval_logger.info("PerformanceAnalyzer initialized with models.")

    def assess_models(self, X_test, y_test):
        """
        Assess the performance of all models on the test set.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True test labels.

        Returns:
            dict: Performance metrics for each model.
        """
        self.performance_metrics = {}
        if not self.model_dict:
            eval_logger.warning("No models provided for assessment.")
            return self.performance_metrics

        for model_name, pipeline in self.model_dict.items():
            eval_logger.info(f"Assessing performance for model: {model_name}")
            try:
                predictions = pipeline.predict(X_test)
                probabilities = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

                metrics = {
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(y_test, predictions, zero_division=0),
                    "recall": recall_score(y_test, predictions, zero_division=0),
                    "f1": f1_score(y_test, predictions, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, probabilities) if probabilities is not None else None,
                    "conf_matrix": confusion_matrix(y_test, predictions),
                    "class_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
                    "predictions": predictions,
                    "probabilities": probabilities
                }
                self.performance_metrics[model_name] = metrics
                eval_logger.info(f"Performance assessment completed for {model_name}.")
            except Exception as e:
                eval_logger.error(f"Failed to assess {model_name}: {str(e)}")
                self.performance_metrics[model_name] = {"error": str(e)}

        return self.performance_metrics

    def display_metrics(self):
        """Display the performance metrics in a formatted manner."""
        if not self.performance_metrics:
            print("No performance metrics to display.")
            return

        print("\n===== Model Performance Summary =====")
        for name, metrics in self.performance_metrics.items():
            if "error" in metrics:
                print(f"\n{name}: Error - {metrics['error']}")
                continue

            print(f"\nModel: {name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "ROC AUC: Not Available")
            print("Confusion Matrix:\n", metrics['conf_matrix'])
        print("=====================================")

    def visualize_roc(self, y_test, output_dir="visuals", show=False):
        """
        Generate ROC curves for all models.

        Args:
            y_test (pd.Series): True test labels.
            output_dir (str): Directory to save the plot.
            show (bool): If True, display the plot interactively.
        """
        if not self.performance_metrics:
            eval_logger.warning("No metrics available to visualize ROC curves.")
            return

        plt.figure(figsize=(10, 7))
        for name, metrics in self.performance_metrics.items():
            if "error" in metrics or metrics.get("probabilities") is None:
                eval_logger.warning(f"Cannot plot ROC for {name}: No probabilities or error occurred.")
                continue
            fpr, tpr, _ = roc_curve(y_test, metrics["probabilities"])
            auc_score = metrics["roc_auc"]
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

        plt.plot([0, 1], [0, 1], "k--", label="Baseline (AUC = 0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Models")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.7)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "roc_comparison.png")
        try:
            plt.savefig(output_path)
            eval_logger.info(f"ROC curves saved to {output_path}")
            if show:
                plt.show()
        except Exception as e:
            eval_logger.error(f"Error saving ROC plot: {str(e)}")
        finally:
            plt.close()

    def visualize_importances(self, output_dir="visuals", max_features=20, show=False):
        """
        Visualize feature importances for tree-based models.

        Args:
            output_dir (str): Directory to save the plots.
            max_features (int): Maximum number of features to display.
            show (bool): If True, display the plot interactively.
        """
        for name, pipeline in self.model_dict.items():
            # Check if the pipeline has a "classifier" step
            if "classifier" not in pipeline.named_steps:
                eval_logger.warning(f"Pipeline for {name} does not have a 'classifier' step. Skipping.")
                continue

            classifier = pipeline.named_steps["classifier"]
            if hasattr(classifier, "feature_importances_"):
                eval_logger.info(f"Generating feature importance plot for {name}...")
                try:
                    features = pipeline.named_steps["preprocessor"].get_feature_names_out()
                    importances = classifier.feature_importances_
                    indices = np.argsort(importances)[-max_features:]

                    plt.figure(figsize=(10, max(5, max_features // 3)))
                    plt.barh(range(len(indices)), importances[indices], align="center")
                    plt.yticks(range(len(indices)), [features[i] for i in indices])
                    plt.xlabel("Importance Score")
                    plt.title(f"Top {max_features} Feature Importances - {name}")
                    plt.gca().invert_yaxis()

                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"importances_{name}.png")
                    plt.savefig(output_path)
                    eval_logger.info(f"Feature importances saved to {output_path}")
                    if show:
                        plt.show()
                except Exception as e:
                    eval_logger.error(f"Failed to plot feature importances for {name}: {str(e)}")
                finally:
                    plt.close()
            else:
                eval_logger.info(f"Feature importances not supported for {name}.")

    def visualize_classification(self, X_test, y_test, model_name, feat_x=None, feat_y=None, output_dir="visuals", show=False):
        """
        Visualize classification results using two features.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels.
            model_name (str): Name of the model to visualize.
            feat_x (str, optional): Feature for x-axis.
            feat_y (str, optional): Feature for y-axis.
            output_dir (str): Directory to save the plot.
            show (bool): If True, display the plot interactively.
        """
        if model_name not in self.performance_metrics or "error" in self.performance_metrics[model_name]:
            eval_logger.warning(f"No results for {model_name}. Skipping visualization.")
            return

        predictions = self.performance_metrics[model_name].get("predictions")
        if predictions is None:
            eval_logger.warning(f"No predictions available for {model_name}.")
            return

        numeric_cols = X_test.select_dtypes(include=np.number).columns.tolist()
        if not feat_x:
            feat_x = numeric_cols[0] if numeric_cols else None
        if not feat_y:
            feat_y = numeric_cols[1] if len(numeric_cols) > 1 else None

        if not feat_x or not feat_y or feat_x == feat_y:
            eval_logger.warning(f"Cannot visualize classification for {model_name}: Invalid features ({feat_x}, {feat_y}).")
            return

        plot_data = pd.DataFrame({
            "Feature_X": X_test[feat_x],
            "Feature_Y": X_test[feat_y],
            "Actual": y_test,
            "Predicted": predictions,
            "Is_Correct": y_test == predictions
        })

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=plot_data, x="Feature_X", y="Feature_Y", hue="Actual",
                        style="Is_Correct", size="Is_Correct", sizes=(50, 150), alpha=0.8)
        plt.title(f"Classification Results for {model_name}\n({feat_x} vs {feat_y})")
        plt.xlabel(feat_x)
        plt.ylabel(feat_y)
        plt.grid(True, linestyle="--", alpha=0.6)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"classification_{model_name}.png")
        try:
            plt.savefig(output_path)
            eval_logger.info(f"Classification plot for {model_name} saved to {output_path}")
            if show:
                plt.show()
        except Exception as e:
            eval_logger.error(f"Failed to save classification plot: {str(e)}")
        finally:
            plt.close()