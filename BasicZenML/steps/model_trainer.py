from zenml.steps import step
from sklearn.ensemble import RandomForestClassifier
import numpy as np
@step
def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
