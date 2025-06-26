import logging

import pandas as pd
from zenml import step

@step
def preprocess_data(df: pd.DataFrame) -> None:
    pass
