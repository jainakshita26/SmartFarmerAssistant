import os
from pymongo import MongoClient
import certifi

# MongoDB connection string
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# Initialize MongoDB Client
# Atlas/SRV connections need TLS; a local mongod usually does not.
client_kwargs = {}
if MONGO_URI.startswith("mongodb+srv://") or "tls=true" in MONGO_URI.lower():
    client_kwargs["tlsCAFile"] = certifi.where()

client = MongoClient(MONGO_URI, **client_kwargs)
db = client.smartcrop

def get_db():
    return db
