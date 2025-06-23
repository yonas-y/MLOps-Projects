"""
Contains Configuration (DB URI, names, paths).
"""

# app/config.py
MONGO_URI = "mongodb://192.168.80.1:27017"  # The address to the local MongoDB!
DB_NAME = "Customers-Satisfaction"
COLLECTION_NAME = "Customers"
MODEL_DIR = "output"
# MODEL_PATH = f"{MODEL_DIR}/ocsvm_model.pkl"
# SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"