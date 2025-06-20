from zenml import pipeline
from steps.data_loader import import_digits_data, show_shapes
from steps.model_trainer import train_model
from steps.evaluator import evaluate_model

@pipeline
def training_pipeline():
    x_train, y_train, x_test, y_test = import_digits_data()
    show_shapes(x_train, y_train, x_test, y_test)
    model = train_model(x_train, y_train)
    training_accuracy = evaluate_model(model, x_train, y_train)
    test_accuracy = evaluate_model(model, x_test, y_test)