# database.py

from pymongo import MongoClient
from bson.objectid import ObjectId
from config import SETTINGS

# Initialize MongoDB client
client = MongoClient(SETTINGS.MONGO_URI)
db = client["StockSim"]

# Collections
users_collection = db["users"]
transactions_collection = db["transactions"]
models_collection = db["models"]
portfolio_snapshots_collection = db["portfolio_snapshots"]
