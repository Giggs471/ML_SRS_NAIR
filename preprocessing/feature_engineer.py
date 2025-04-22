import pandas as pd
from logging import getLogger, INFO, Formatter, StreamHandler

# Logger setup
feat_logger = getLogger("FeatureCreator")
feat_logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
feat_logger.addHandler(handler)

class FeatureCreator:
    """Generates new features for user profile data."""

    def __init__(self):
        """Initialize the feature creator."""
        feat_logger.info("FeatureCreator initialized.")

    def calculate_completeness_score(self, df):
        """
        Calculate a weighted completeness score based on profile attributes.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with a 'completeness_score' column.
        """
        df_new = df.copy()
        # Identify boolean-like columns (0/1, True/False)
        bool_cols = [col for col in df_new.columns if set(df_new[col].dropna().unique()).issubset({0, 1, True, False, 0.0, 1.0})]
        feat_logger.info(f"Boolean columns found for completeness scoring: {bool_cols}")

        if not bool_cols:
            feat_logger.warning("No boolean columns found for completeness scoring.")
            df_new["completeness_score"] = 0.0
            return df_new

        # Categorize columns into high and low importance
        high_importance = [col for col in bool_cols if any(kw in col.lower() for kw in ["name", "verified", "occupation"])]
        low_importance = [col for col in bool_cols if col not in high_importance]
        feat_logger.info(f"High importance columns: {high_importance}")
        feat_logger.info(f"Low importance columns: {low_importance}")

        # Convert to numeric and fill NaNs
        for col in bool_cols:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(0)

        # Weighted scoring: high importance cols get weight 2, others get weight 1
        high_weight = 2
        low_weight = 1
        total_weight = (len(high_importance) * high_weight) + (len(low_importance) * low_weight)
        if total_weight == 0:
            df_new["completeness_score"] = 0.0
        else:
            score = (df_new[high_importance].sum(axis=1) * high_weight + df_new[low_importance].sum(axis=1) * low_weight) / total_weight
            df_new["completeness_score"] = score

        feat_logger.info(f"Calculated completeness_score using {len(bool_cols)} columns.")
        return df_new

    def calculate_privacy_score(self, df):
        """
        Calculate a privacy score based on visibility-related features.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with a 'privacy_score' column.
        """
        df_new = df.copy()
        privacy_cols = [col for col in df_new.columns if "visible" in col.lower() or "closed" in col.lower()]
        feat_logger.info(f"Privacy-related columns found: {privacy_cols}")
        
        if not privacy_cols:
            feat_logger.warning("No privacy-related columns found.")
            df_new["privacy_score"] = 0.0
            return df_new

        for col in privacy_cols:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(0)

        df_new["privacy_score"] = df_new[privacy_cols].mean(axis=1)
        feat_logger.info(f"Calculated privacy_score using {len(privacy_cols)} columns.")
        return df_new

    def generate_features(self, df):
        """
        Generate all new features.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new features.
        """
        feat_logger.info("Generating new features...")
        feat_logger.info(f"Input columns: {df.columns.tolist()}")
        df_new = self.calculate_completeness_score(df)
        df_new = self.calculate_privacy_score(df_new)
        feat_logger.info("Feature generation completed.")
        feat_logger.info(f"Output columns after feature engineering: {df_new.columns.tolist()}")
        return df_new