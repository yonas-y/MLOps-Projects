from zenml import pipeline
from steps.data_loader import import_digits_data, show_shapes
from steps.model_trainer import train_model

@pipeline
def training_pipeline():
    x_train, y_train, x_test, y_test = import_digits_data()
    show_shapes(x_train, y_train, x_test, y_test)
    model = train_model(x_train, y_train)


# @pipeline
# def training_pipeline(import_data, train_model, evaluate_model):
#     X, y = import_data()
#     model = train_model(X=X, y=y)
#     evaluate_model(model=model, X=X, y=y)