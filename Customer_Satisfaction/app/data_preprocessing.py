import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib


class DataPreprocessing:
    """
    A class used for cleaning and preprocessing customer satisfaction data.

    Methods
    -------
    preprocess(df: pd.DataFrame) -> pd.DataFrame
        Perform preprocessing steps like handling missing values, dropping duplicates, etc.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): The raw input data.
        """
        self.raw_data = dataframe
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()

    def data_cleaning(self) -> pd.DataFrame:
        """
        Cleans the data by:
        - Selecting only important columns used for preprocessing.
        - Filling missing values with column medians

        Returns:
            pd.DataFrame: The cleaned data.
        """
        df = self.raw_data.copy()
        if df.empty:
            raise logging.error("❌ No data was loaded from MongoDB.")

        try:
            df = df.drop_duplicates()
            df = df.dropna(how='all')

            # Separate the numeric df!
            numeric_df = df.select_dtypes(include="number")
            df_ord_status = df["order_status"]  # Using only the order status as a non-numeric feature!

            df_combined = pd.concat([numeric_df, df_ord_status], axis=1)

            # # Else we can drop selected features which are not required for modeling!
            # cols_to_drop = ["order_id", "customer_id", "product_category_name", "customer_unique_id",
            #                 "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date"]
            # data_df = df.drop(cols_to_drop, axis=1)
            # data_df["review_comment_message"] = data_df["review_comment_message"].replace('', pd.NA).fillna("No review!")

            # Fill NaNs with median for numeric columns only
            medians = df_combined.median(numeric_only=True)
            df_combined.fillna(medians, inplace=True)

        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

        return df_combined
