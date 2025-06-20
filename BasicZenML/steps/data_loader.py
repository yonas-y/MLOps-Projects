from zenml import step
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from typing import Tuple
import logging
import numpy as np

@step
def import_digits_data() -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_test"]
]:
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test

logger = logging.getLogger(__name__)
@step
def show_shapes(x_train, y_train, x_test, y_test) -> None:
    logger.info(f"x_train: {x_train.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"x_test: {x_test.shape}")
    logger.info(f"y_test: {y_test.shape}")
