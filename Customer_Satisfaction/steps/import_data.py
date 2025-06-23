from zenml import steps
from typing import Tuple
import logging
import pandas as pd
from Customer_Satisfaction.app.load_data import load_data_from_mongodb

logger = logging.getLogger(__name__)

@step
def import_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    A ZenML step that imports data from MongoDB and splits it into features and labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature DataFrame 'X'
        and the target Series 'y'.

    Raises:
        ValueError: If the DataFrame is empty (no data loaded from MongoDB).
    """
    df = load_data_from_mongodb()

    if df.empty:
        raise logging.error("❌ No data was loaded from MongoDB.")

    logger.info('Getting the X and y data from the df!')
    X = df.drop(columns=["review_score"])
    y = df["review_score"]

    return X, y
