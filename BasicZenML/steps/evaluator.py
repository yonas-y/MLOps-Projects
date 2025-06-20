from zenml.steps import step
from sklearn.metrics import accuracy_score
import numpy as np

@step
def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc}")
    return acc
