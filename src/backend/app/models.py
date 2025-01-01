# models.py

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime, timedelta
import os
import shutil
import json
import asyncio
import numpy as np
import pandas as pd
import logging

from config import SETTINGS
from database import models_collection
from auth import get_current_user

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

router = APIRouter(
    prefix="/api",
    tags=["Model Management"]
)

# Logger setup
logger = logging.getLogger(__name__)

# Utility Functions
async def get_stock_training_data(symbol: str, period: str = "2y"):
    """Fetch and preprocess stock data for training"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'] = calculate_macd(df['Close'])
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

def prepare_sequences(data, sequence_length=60):
    """Prepare sequences for LSTM training"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predicting the closing price
    
    return np.array(X), np.array(y), scaler

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    """Calculate MACD technical indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

async def event_generator(model_id: str):
    """Generator for SSE events to track model training status."""
    try:
        while True:
            model_info = models_collection.find_one({"id": model_id})
            if not model_info:
                yield json.dumps({"error": "Model not found"})
                return
                
            yield json.dumps({
                "status": model_info.get("status"),
                "accuracy": model_info.get("accuracy"),
                "test_loss": model_info.get("test_loss")
            })
            
            if model_info.get("status") == "completed":
                break
                
            await asyncio.sleep(1)
    except Exception as e:
        yield json.dumps({"error": str(e)})

# Endpoints

@router.post("/model/train")
async def train_model(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    try:
        body = await request.json()
        symbol = body["symbol"].upper()
        config = body.get("config", {
            "epochs": 50,
            "batch_size": 32,
            "sequence_length": 60,
            "learning_rate": 0.001
        })
        
        # Fetch and prepare data
        df = await get_stock_training_data(symbol)
        
        # Prepare features
        features = np.column_stack((
            df['Close'],
            df['Volume'],
            df['MA5'],
            df['MA20'],
            df['RSI'],
            df['MACD']
        ))
        
        # Prepare sequences
        X, y, scaler = prepare_sequences(features, config['sequence_length'])
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build model
        model = Sequential([
            LSTM(64, input_shape=(config['sequence_length'], X.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse'
        )
        
        # Save model metadata
        model_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_info = {
            "id": model_id,
            "user_email": current_user,
            "symbol": symbol,
            "created_at": datetime.now(timezone.utc),
            "config": config,
            "status": "training"
        }
        models_collection.insert_one(model_info)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Save model
        model_path = os.path.join(SETTINGS.MODEL_DIR, model_id)
        model.save(model_path)
        
        # Calculate accuracy metrics
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        accuracy = 1 - np.mean(np.abs((y_test_actual - predictions) / y_test_actual))
        
        # Update model info
        models_collection.update_one(
            {"id": model_id},
            {
                "$set": {
                    "status": "completed",
                    "accuracy": float(accuracy),
                    "test_loss": float(test_loss)
                }
            }
        )
        
        return {
            "model_id": model_id,
            "accuracy": float(accuracy),
            "test_loss": float(test_loss)
        }
        
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/training-status/{model_id}")
async def model_training_status(
    model_id: str,
    current_user: str = Depends(get_current_user)
):
    from sse_starlette.sse import EventSourceResponse
    return EventSourceResponse(event_generator(model_id))

@router.post("/model/predict")
async def predict(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    try:
        body = await request.json()
        model_id = body["model_id"]
        symbol = body["symbol"].upper()
        
        # Load model
        model_path = os.path.join(SETTINGS.MODEL_DIR, model_id)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = load_model(model_path)
        
        # Fetch recent data
        df = await get_stock_training_data(symbol, period="60d")
        
        # Prepare features
        features = np.column_stack((
            df['Close'],
            df['Volume'],
            df['MA5'],
            df['MA20'],
            df['RSI'],
            df['MACD']
        ))
        
        # Scale and prepare sequence
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features)
        sequence_length = int(body.get("sequence_length", 60))
        sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, features.shape[1])
        
        # Make prediction
        prediction = model.predict(sequence)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        
        return {
            "symbol": symbol,
            "prediction": float(prediction[0][0]),
            "current_price": float(df['Close'].iloc[-1]),
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_models(current_user: str = Depends(get_current_user)):
    try:
        models = list(models_collection.find({"user_email": current_user}))
        
        for model in models:
            # Convert ObjectId to string
            if "_id" in model:
                model["id"] = str(model["_id"])
                del model["_id"]  # Optional: Remove the _id field if not needed
            else:
                # Handle cases where _id might not be present
                model["id"] = None
        
        return {"models": models}
    except Exception as e:
        logger.error(f"Get models error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/upload")
async def upload_model(
    model: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    try:
        # Validate file extension
        valid_extensions = ['.h5', '.keras', '.pkl', '.joblib']
        file_extension = os.path.splitext(model.filename)[1].lower()
        
        if file_extension not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported formats: {', '.join(valid_extensions)}"
            )

        # Create unique filename
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{current_user}_{timestamp}{file_extension}"
        file_path = os.path.join(SETTINGS.MODEL_DIR, unique_filename)

        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(model.file, buffer)

        # Create model document
        model_doc = {
            "user_email": current_user,
            "name": model.filename,
            "path": file_path,
            "uploadDate": datetime.now(timezone.utc),
            "status": "inactive",
            "features": ["price", "volume", "technical_indicators"],  # Default features
        }

        # Insert into database
        result = models_collection.insert_one(model_doc)
        model_doc["id"] = str(result.inserted_id)
        del model_doc["_id"]

        return {
            "message": "Model uploaded successfully",
            "model": model_doc
        }

    except Exception as e:
        # Clean up file if it was created
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Model upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/toggle")
async def toggle_model(
    model_id: str,
    current_user: str = Depends(get_current_user)
):
    try:
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
            
        # Find the model and verify ownership
        model = models_collection.find_one({
            "id": model_id,
            "user_email": current_user
        })
        
        if not model:
            raise HTTPException(
                status_code=404, 
                detail=f"Model not found with ID: {model_id}"
            )

        # Toggle the model status
        current_status = model.get("status", "inactive")
        new_status = "active" if current_status != "active" else "inactive"
        
        # Update this model's status
        result = models_collection.update_one(
            {"id": model_id},
            {"$set": {"status": new_status}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to update model status"
            )

        return {
            "status": new_status,
            "model_id": model_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Toggle model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: str = Depends(get_current_user)
):
    try:
        # Find the model
        model = models_collection.find_one({
            "id": model_id,
            "user_email": current_user
        })
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Delete the file if it exists
        if os.path.exists(model["path"]):
            os.remove(model["path"])

        # Delete from database
        models_collection.delete_one({"id": model_id})

        return {"message": "Model deleted successfully"}

    except Exception as e:
        logger.error(f"Delete model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
