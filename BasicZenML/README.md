# 🧠 Basic MLOps Pipeline with ZenML

[![ZenML](https://img.shields.io/badge/MLOps-ZenML-blueviolet?style=flat-square&logo=zenml)](https://zenml.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

This project demonstrates a basic end-to-end MLOps pipeline using [ZenML](https://zenml.io/). The goal is to build a modular and reproducible workflow that:

- Loads digit classification data using `sklearn.datasets.load_digits`
- Splits the data into training and testing sets
- Trains a `RandomForestClassifier`
- Evaluates accuracy on training and test sets
- Visualizes results using ZenML’s dashboard and caching

---

## 📦 Project Structure

```bash
BasicZenML/
├── main.py                       # Entry point to run the pipeline
├── pipelines/
│   └── training_pipeline.py      # ZenML pipeline definition
├── steps/
│   ├── data_loader.py            # Data loading and preprocessing step
│   ├── model_trainer.py          # Model training step
│   └── evaluator.py              # Model evaluation step
├── README.md                     # Documentation
└── requirements.txt              # Python dependencies
