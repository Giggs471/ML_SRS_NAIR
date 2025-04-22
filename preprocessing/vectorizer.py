import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from logging import getLogger, INFO, Formatter, StreamHandler

# Configure logger
vector_logger = getLogger("DataTransformer")
vector_logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
vector_logger.addHandler(handler)

class DataTransformer:
    """Prepares features for machine learning by transforming numerical and categorical data."""

    def __init__(self, num_columns, cat_columns, num_impute_method="mean", cat_impute_method="most_frequent", cat_missing_value="unknown"):
        """
        Initialize the DataTransformer with column specifications.

        Args:
            num_columns (list): List of numerical column names.
            cat_columns (list): List of categorical column names.
            num_impute_method (str): Imputation method for numerical columns.
            cat_impute_method (str): Imputation method for categorical columns.
            cat_missing_value (str): Value to fill missing categorical data.
        """
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.transform_pipeline = None
        self.transformed_feature_names = None
        vector_logger.info("DataTransformer initialized.")

        # Define pipelines for numerical and categorical data
        num_pipeline = Pipeline([
            ("num_impute", SimpleImputer(strategy=num_impute_method)),
            ("num_scale", RobustScaler())  # Using RobustScaler instead of StandardScaler
        ])

        cat_pipeline = Pipeline([
            ("cat_impute", SimpleImputer(strategy=cat_impute_method, fill_value=cat_missing_value)),
            ("cat_encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
        ])

        # Build the ColumnTransformer
        self.transform_pipeline = ColumnTransformer(
            transformers=[
                ("numerical", num_pipeline, self.num_columns),
                ("categorical", cat_pipeline, self.cat_columns)
            ],
            remainder="drop"  # Drop any columns not specified
        )
        vector_logger.debug("ColumnTransformer created with numerical and categorical pipelines.")

    def prepare(self, data):
        """
        Fit the transformer to the provided data.

        Args:
            data (pd.DataFrame): Data to fit the transformer on.
        """
        try:
            # Filter columns to those present in the data
            present_num_cols = [col for col in self.num_columns if col in data.columns]
            present_cat_cols = [col for col in self.cat_columns if col in data.columns]

            if not present_num_cols and not present_cat_cols:
                vector_logger.error("No specified columns found in the data.")
                raise ValueError("No matching columns found for transformation.")

            # Update the transformer's column specifications
            self.transform_pipeline.transformers_ = [
                ("numerical", self.transform_pipeline.transformers[0][1], present_num_cols),
                ("categorical", self.transform_pipeline.transformers[1][1], present_cat_cols)
            ]

            # Fit the transformer
            self.transform_pipeline.fit(data)
            self.transformed_feature_names = self.transform_pipeline.get_feature_names_out()
            vector_logger.info("Transformer fitted successfully.")
        except Exception as e:
            vector_logger.error(f"Failed to fit transformer: {str(e)}")
            raise

    def apply_transform(self, data):
        """
        Apply the fitted transformer to the data.

        Args:
            data (pd.DataFrame): Data to transform.

        Returns:
            np.ndarray: Transformed data, or None if an error occurs.
        """
        if self.transform_pipeline is None:
            vector_logger.error("Transformer not fitted. Call prepare() first.")
            return None

        try:
            transformed_data = self.transform_pipeline.transform(data)
            vector_logger.info(f"Data transformed successfully. Shape: {transformed_data.shape}")
            return transformed_data
        except Exception as e:
            vector_logger.error(f"Transformation failed: {str(e)}")
            return None

    def get_transformed_columns(self):
        """
        Retrieve the names of the transformed features.

        Returns:
            list: Names of the transformed features, or None if not fitted.
        """
        if self.transformed_feature_names is None:
            vector_logger.warning("Transformer not fitted or feature names not available.")
            return None
        return self.transformed_feature_names.tolist()