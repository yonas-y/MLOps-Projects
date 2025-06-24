from pymongo import MongoClient
import pandas as pd
import os
from app.config import HOST_IP, HOST_PORT, DB_NAME, COLLECTION_NAME

def load_data_from_mongodb(
        host: str = HOST_IP, port: int = HOST_PORT,
        db_name: str = DB_NAME, collection_name: str = COLLECTION_NAME
) -> pd.DataFrame:
    """
    Connects to a MongoDB instance and retrieves data from the specified collection.

    Args:
        host (str): The IP address or hostname of the MongoDB server.
        port (int): The port on which MongoDB is running.
        db_name (str): The name of the MongoDB database to query.
        collection_name (str): The name of the collection from which to retrieve documents.

    Returns:
        pd.DataFrame: A DataFrame!
    """

    uri = f"mongodb://{host}:{port}"
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)

    try:
        db = client[db_name]
        collection = db[collection_name]
        data = list(collection.find())
        df = pd.DataFrame(data)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        return df
    except Exception as e:
        print("❌ MongoDB connection or query failed:", e)
