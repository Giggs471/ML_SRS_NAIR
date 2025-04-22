import pandas as pd
import os
from logging import getLogger, INFO, Formatter, StreamHandler

# Set up logger
logger = getLogger("DataReader")
logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
logger.addHandler(handler)

class DataReader:
    """A utility class for reading data from CSV files into a DataFrame."""

    def __init__(self, path_to_file):
        """
        Initialize the DataReader with the path to the CSV file.

        Args:
            path_to_file (str): Path to the CSV file to be loaded.
        """
        self.path_to_file = path_to_file
        self.df = None
        logger.info(f"DataReader initialized with path: {self.path_to_file}")

    def read_csv_data(self, missing_values=["Unknown", ""]):
        """
        Read data from the specified CSV file.

        Args:
            missing_values (list): Values to treat as NaN during loading.

        Returns:
            pd.DataFrame: The loaded DataFrame, or None if an error occurs.
        """
        # Check if file exists
        if not os.path.exists(self.path_to_file):
            logger.error(f"CSV file not found at: {self.path_to_file}")
            return None

        try:
            # Load the CSV with specified NaN values
            self.df = pd.read_csv(self.path_to_file, na_values=missing_values)
            logger.info(f"Data loaded successfully from {self.path_to_file}. Shape: {self.df.shape}")

            # Drop any columns that start with "Unnamed"
            unnamed_cols = [col for col in self.df.columns if col.startswith("Unnamed")]
            if unnamed_cols:
                self.df.drop(columns=unnamed_cols, inplace=True)
                logger.info(f"Dropped {len(unnamed_cols)} unnamed columns. New shape: {self.df.shape}")

            return self.df

        except Exception as e:
            logger.error(f"Failed to load data from {self.path_to_file}: {str(e)}")
            return None