from zenml.steps import step
from sklearn.metrics import accuracy_score

@step
def evaluate_model(model, X: list, y: list) -> float:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc}")
    return acc
