from zenml.pipelines import pipeline

@pipeline
def training_pipeline(import_data, train_model, evaluate_model):
    X, y = import_data()
    model = train_model(X=X, y=y)
    evaluate_model(model=model, X=X, y=y)
