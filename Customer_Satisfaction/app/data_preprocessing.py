import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os


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
        self.target_column = "review_score"
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

    def data_encoding(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the non-numeric values of the cleaned data.

        Args:
            cleaned_data (pd.DataFrame): The cleaned data.

        Returns:
            pd.DataFrame: The encoded data.
        """
        # Separate numeric and non-numeric
        numeric_df = cleaned_data.select_dtypes(include="number")
        non_numeric_df = cleaned_data.select_dtypes(exclude="number")

        # One-hot encode non-numeric
        encoded_array = self.encoder.fit_transform(non_numeric_df)
        encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out(non_numeric_df.columns))

        # Combine both
        encoded_final_df = pd.concat([numeric_df, encoded_df], axis=1)

        return encoded_final_df

    def data_split(self, encoded_data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the cleaned data into training and testing sets.

        Args:
            encoded_data (pd.DataFrame): The cleaned and encoded data.
            test_size (float): Fraction of the data to be used as test set.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """

        X = encoded_data.drop(columns=[self.target_column])
        y = encoded_data[self.target_column]

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def data_normalization(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Normalizes the training and test data!.

        Args:
            X_train (pd.DataFrame): The training data.
            X_test (pd.DataFrame): The test data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: X_train_normalized, X_test_normalized
        """
        X_train_scaled = self.scaler.fit_transform(X_train)  # fit on train
        X_test_scaled = self.scaler.transform(X_test)

        os.makedirs("output", exist_ok=True)
        joblib.dump(self.scaler, 'output/scaler.pkl')  # save

        return X_train_scaled, X_test_scaled
