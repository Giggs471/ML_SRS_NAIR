import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from logging import getLogger, INFO, Formatter, StreamHandler

# Logger setup
cleaner_logger = getLogger("DataProcessor")
cleaner_logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
cleaner_logger.addHandler(handler)

class DataProcessor:
    """Processes and cleans raw data for machine learning."""

    def __init__(self):
        """Initialize the data processor."""
        self.num_imputer = None
        self.cat_imputer = None
        cleaner_logger.info("DataProcessor initialized.")

    def detect_column_types(self, df, target_col):
        """
        Detect numerical and categorical columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Target column name.

        Returns:
            tuple: Lists of numerical and categorical columns.
        """
        # Initial detection based on dtype
        num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Refine numerical columns: exclude columns with only 0/1 values (treat as categorical)
        true_num_cols = []
        for col in num_cols:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2:
                cat_cols.append(col)  # Treat as categorical
            else:
                true_num_cols.append(col)

        num_cols = true_num_cols

        # Remove target column from feature lists
        if target_col in num_cols:
            num_cols.remove(target_col)
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        cleaner_logger.info(f"Detected {len(num_cols)} numerical columns: {num_cols}")
        cleaner_logger.info(f"Detected {len(cat_cols)} categorical columns: {cat_cols}")
        return num_cols, cat_cols

    def initialize_imputers(self, df, num_cols, cat_cols, num_method="knn", cat_method="most_frequent"):
        """
        Initialize imputers for numerical and categorical columns.

        Args:
            df (pd.DataFrame): DataFrame to fit imputers on.
            num_cols (list): List of numerical columns.
            cat_cols (list): List of categorical columns.
            num_method (str): Imputation method for numerical ("knn" or "median").
            cat_method (str): Imputation method for categorical.
        """
        if num_cols:
            if num_method == "knn":
                self.num_imputer = KNNImputer(n_neighbors=5)
            else:
                self.num_imputer = SimpleImputer(strategy="median")
            self.num_imputer.fit(df[num_cols])
            cleaner_logger.info(f"Initialized numerical imputer with method: {num_method}")
        else:
            cleaner_logger.warning("No numerical columns to initialize imputer for.")

        if cat_cols:
            self.cat_imputer = SimpleImputer(strategy=cat_method, fill_value="missing")
            self.cat_imputer.fit(df[cat_cols])
            cleaner_logger.info(f"Initialized categorical imputer with method: {cat_method}")
        else:
            cleaner_logger.warning("No categorical columns to initialize imputer for.")

    def fill_missing(self, df, num_cols, cat_cols):
        """
        Fill missing values in the DataFrame using fitted imputers.

        Args:
            df (pd.DataFrame): DataFrame to process.
            num_cols (list): List of numerical columns.
            cat_cols (list): List of categorical columns.

        Returns:
            pd.DataFrame: DataFrame with missing values filled.
        """
        df_processed = df.copy()
        if self.num_imputer and num_cols:
            present_num_cols = [col for col in num_cols if col in df_processed.columns]
            if present_num_cols:
                df_processed[present_num_cols] = self.num_imputer.transform(df_processed[present_num_cols])
                cleaner_logger.info(f"Filled missing values in {len(present_num_cols)} numerical columns.")
            else:
                cleaner_logger.warning("No matching numerical columns for imputation.")

        if self.cat_imputer and cat_cols:
            present_cat_cols = [col for col in cat_cols if col in df_processed.columns]
            if present_cat_cols:
                df_processed[present_cat_cols] = self.cat_imputer.transform(df_processed[present_cat_cols])
                cleaner_logger.info(f"Filled missing values in {len(present_cat_cols)} categorical columns.")
            else:
                cleaner_logger.warning("No matching categorical columns for imputation.")

        return df_processed

    def remove_duplicates(self, df):
        """
        Remove duplicate rows from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        original_size = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = original_size - len(df_cleaned)
        cleaner_logger.info(f"Removed {duplicates_removed} duplicate rows. New size: {len(df_cleaned)}")
        return df_cleaned

    def process_data(self, df, target_col):
        """
        Apply all data processing steps.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Target column name.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        cleaner_logger.info("Starting data processing...")
        df_processed = self.remove_duplicates(df)
        num_cols, cat_cols = self.detect_column_types(df_processed, target_col)
        if self.num_imputer or self.cat_imputer:
            df_processed = self.fill_missing(df_processed, num_cols, cat_cols)
        else:
            cleaner_logger.warning("Imputers not initialized. Ensure initialize_imputers() is called on training data.")
        cleaner_logger.info("Data processing completed.")
        return df_processed