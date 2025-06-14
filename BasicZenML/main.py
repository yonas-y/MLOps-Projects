from pipelines.training_pipeline import training_pipeline
from steps.data_loader import import_data
from steps.model_trainer import train_model
from steps.evaluator import evaluate_model

if __name__ == "__main__":
    training_pipeline(
        import_data=import_data(),
        train_model=train_model(),
        evaluate_model=evaluate_model()
    ).run()
