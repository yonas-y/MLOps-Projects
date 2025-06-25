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
