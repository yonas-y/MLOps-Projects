import numpy as np
import pandas as pd

import logging
from zenml import step

@step
def model_training(df: pd.DataFrame) -> None:
    pass
