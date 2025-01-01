# Update in config.py

import os
from dotenv import load_dotenv
from pydantic import BaseSettings
from pathlib import Path

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings(BaseSettings):
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-default-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MODEL_DIR: str = os.path.join(BASE_DIR, "models")
    ALLOW_ORIGINS: list = ["http://localhost:3000"]
    HOST: str = "0.0.0.0"
    PORT: int = 5000

    class Config:
        env_file = ".env"

# Instantiate settings
SETTINGS = Settings()

# Ensure the models directory exists
os.makedirs(SETTINGS.MODEL_DIR, exist_ok=True)
print(f"Models directory created/verified at: {SETTINGS.MODEL_DIR}")