from zenml import pipeline
from steps.import_data import import_data
from steps.clean_data import clean_data
from steps.model_training import model_training
from steps.evaluation import evaluation

@pipeline
def training_pipeline():
    x_data, y_data = import_data()
    clean_data(x_data)
    model_training(x_data)
    evaluation(x_data, y_data)
