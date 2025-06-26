from zenml import step
import logging

import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from Customer_Satisfaction.app.data_preprocessing import DataPreprocessing

@step
def preprocess_data(data: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_test"]
]:
    data_preprocessed = DataPreprocessing(data)
    cleaned_data = data_preprocessed.data_cleaning()
    encoded_data = data_preprocessed.data_encoding(cleaned_data)
    X_train, X_test, y_train, y_test = data_preprocessed.data_split(encoded_data)
    X_train_scaled, X_test_scale = data_preprocessed.data_normalization(X_train=X_train, X_test=X_test)

    return X_train_scaled, y_train, X_test_scale, y_test
