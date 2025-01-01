# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config import SETTINGS
from database import db
import auth
import stock
import models

# Initialize FastAPI app
app = FastAPI(title="Stock API with Auth", version="1.0.1")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Include routers from other modules
app.include_router(auth.router)
app.include_router(stock.router)
app.include_router(models.router)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock API with Auth"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SETTINGS.HOST, port=SETTINGS.PORT)
