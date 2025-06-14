from zenml.steps import step
from sklearn.ensemble import RandomForestClassifier

@step
def train_model(X: list, y: list) -> RandomForestClassifier:
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
