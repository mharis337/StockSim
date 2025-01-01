from pymongo import MongoClient
from bson.objectid import ObjectId
from config import SETTINGS

client = MongoClient(SETTINGS.MONGO_URI)
db = client["StockSim"]

users_collection = db["users"]
transactions_collection = db["transactions"]
models_collection = db["models"]
portfolio_snapshots_collection = db["portfolio_snapshots"]
