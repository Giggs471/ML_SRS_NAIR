from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from logging import getLogger, INFO, Formatter, StreamHandler

# Logger setup
trainer_logger = getLogger("ModelBuilder")
trainer_logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
trainer_logger.addHandler(handler)

class ModelBuilder:
    """Class for building and training classification models."""

    def __init__(self, model_types=["logistic", "rf", "gb"], seed=42):
        """
        Initialize with a list of model types to train.

        Args:
            model_types (list): List of model types ("logistic", "rf", "gb").
            seed (int): Random seed for reproducibility.
        """
        self.model_types = model_types
        self.seed = seed
        self.trained_models = {}
        trainer_logger.info(f"ModelBuilder initialized for: {self.model_types}")

    def _create_model(self, model_type):
        """Create a model instance based on the type."""
        if model_type == "logistic":
            return LogisticRegression(random_state=self.seed, max_iter=1000, class_weight="balanced")
        elif model_type == "rf":
            return RandomForestClassifier(random_state=self.seed, class_weight="balanced")
        elif model_type == "gb":
            return GradientBoostingClassifier(random_state=self.seed)
        else:
            trainer_logger.warning(f"Unsupported model type: {model_type}")
            return None

    def build_and_train(self, preprocessor, X_train, y_train):
        """
        Build and train models using a preprocessing pipeline.

        Args:
            preprocessor: The preprocessing pipeline.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            dict: Dictionary of trained pipelines.
        """
        self.trained_models = {}
        for model_type in self.model_types:
            model = self._create_model(model_type)
            if model is None:
                continue

            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("model", model)
            ])
            try:
                trainer_logger.info(f"Building and training {model_type} model...")
                pipeline.fit(X_train, y_train)
                self.trained_models[model_type] = pipeline
                trainer_logger.info(f"{model_type} model training completed.")
            except Exception as e:
                trainer_logger.error(f"Training failed for {model_type}: {str(e)}")
        return self.trained_models

    def get_models(self):
        """Return the dictionary of trained models."""
        return self.trained_models