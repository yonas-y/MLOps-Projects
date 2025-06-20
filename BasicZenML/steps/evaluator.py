from zenml.steps import step
from sklearn.metrics import accuracy_score
import numpy as np
import logging

logger = logging.getLogger(__name__)
@step
def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    logger.info(f"Accuracy: {acc}")
    return acc
