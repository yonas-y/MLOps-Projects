from zenml import pipeline
from Customer_Satisfaction.steps.import_data import import_data
from Customer_Satisfaction.steps.clean_data import clean_data
from Customer_Satisfaction.steps.model_training import model_training
from Customer_Satisfaction.steps.evaluation import evaluation

@pipeline
def training_pipeline():
    x_data, y_data = import_data()
    clean_data(x_data)
    clean_data(y_data)
    model_training(x_data)
    evaluation(x_data, y_data)
