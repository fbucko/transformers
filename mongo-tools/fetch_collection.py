"""
@file fetch_collection.py

@author Filip Bucko
@date 2023-10-05
@brief This script connects to a MongoDB database, fetches a specified collection, and converts the data into a Parquet file format.
"""
import os
import pandas as pd
from pymongo import MongoClient

username = os.getenv("MONGO_USER")
password = os.getenv("MONGO_PASS")

connection_string = f"mongodb://{username}:{password}@feta3.fit.vutbr.cz:27017"
client = MongoClient(connection_string)

# # Access the database (if not specified in the connection string)
db = client.get_database("drdb")

# # Access the desired collection

collection = db.benign_2312_anonymized_HTML

# Fetch all documents from the collection
documents = list(collection.find())
print(documents[:10])

# # Convert to DataFrame
df = pd.DataFrame(documents)

# Drop MongoDB's default _id field
if '_id' in df.columns:
    df.drop('_id', axis=1, inplace=True)

# # Write the DataFrame to a Parquet file
df.to_parquet('output.parquet', engine='pyarrow', index=False)
print("Data has been successfully exported to output.parquet")