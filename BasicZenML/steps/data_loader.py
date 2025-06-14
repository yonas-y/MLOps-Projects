from zenml.steps import step
from sklearn.datasets import load_iris

@step
def import_data():
    data = load_iris()
    return data.data, data.target
