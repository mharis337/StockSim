# models.py

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime, timedelta, timezone
import os
import shutil
import json
import asyncio
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import yfinance as yf
from model_analyzer import ModelAnalyzer
from model_cache import ModelCacheManager, model_cache

from config import SETTINGS
from database import models_collection
from auth import get_current_user

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout,TFSMLayer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import talib

router = APIRouter(
    prefix="/api",
    tags=["Model Management"]
)

# Logger setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Ensure models directory exists
if not os.path.exists(SETTINGS.MODEL_DIR):
    os.makedirs(SETTINGS.MODEL_DIR, exist_ok=True)
    logger.info(f"Created models directory at: {SETTINGS.MODEL_DIR}")
else:
    logger.info(f"Using existing models directory at: {SETTINGS.MODEL_DIR}")




# Utility Functions
async def get_stock_training_data(symbol: str, period: str = "1y"):
    """Fetch and preprocess stock data for training"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Convert columns to double precision
        df = df.astype('float64')
        
        # Calculate technical indicators
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Price based indicators
        try:
            df['MA5'] = talib.SMA(close, timeperiod=5)
            df['MA20'] = talib.SMA(close, timeperiod=20)
            df['EMA20'] = talib.EMA(close, timeperiod=20)
            
            # Momentum indicators
            df['RSI'] = talib.RSI(close)
            df['ROC'] = talib.ROC(close, timeperiod=10)
            
            # Volatility indicators
            df['ATR'] = talib.ATR(high, low, close)
            df['BBANDS_U'], df['BBANDS_M'], df['BBANDS_L'] = talib.BBANDS(close)
            
            # Volume indicators
            df['OBV'] = talib.OBV(close, volume)
            df['AD'] = talib.AD(high, low, close, volume)
            df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # Trend indicators
            df['ADX'] = talib.ADX(high, low, close)
            df['PLUS_DI'] = talib.PLUS_DI(high, low, close)
            df['MINUS_DI'] = talib.MINUS_DI(high, low, close)
            df['MACD'], df['MACD_SIGNAL'], _ = talib.MACD(close)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise ValueError(f"Error calculating indicators: {str(e)}")
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
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


def calculate_target_prices(current_price, predicted_price):
    """Calculate target prices and confidence for signals"""
    predicted_change = (predicted_price - current_price) / current_price
    
    # Calculate confidence based on the magnitude of predicted change
    # Scale it to be between 0-100
    confidence = min(abs(predicted_change) * 200, 100)  # 50% change = 100% confidence
    
    # Set target price based on predicted movement
    if predicted_change > 0:  # Buy signal
        target_price = predicted_price
        stop_loss = current_price * 0.98  # 2% stop loss
    else:  # Sell signal
        target_price = predicted_price
        stop_loss = current_price * 1.02  # 2% stop above entry
    
    return target_price, stop_loss, confidence

# Endpoints

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Prepare feature set for prediction"""
    feature_columns = [
        'Close', 'Volume', 
        'MA5', 'MA20', 'EMA20',
        'RSI', 'ROC',
        'ATR', 'BBANDS_U', 'BBANDS_M', 'BBANDS_L',
        'OBV', 'AD', 'MFI',
        'ADX', 'PLUS_DI', 'MINUS_DI',
        'MACD', 'MACD_SIGNAL'
    ]
    
    # Make sure all values are float64
    for col in feature_columns:
        df[col] = df[col].astype('float64')
        
    return df[feature_columns].values

@router.post("/model/{model_id}/analyze")
async def analyze_stock(
    model_id: str,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    try:
        body = await request.json()
        symbol = body["symbol"].upper()
        selected_features = body.get("features", [])
        
        logger.info(f"Analyzing {symbol} with features: {selected_features}")
        
        # Get model info from database
        model_info = models_collection.find_one({
            "id": model_id,
            "user_email": current_user
        })
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        model_path = model_info.get('path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="Model file not found")

        # Initialize analyzer
        analyzer = ModelAnalyzer(model_path)
        
        # Get stock data
        df = await get_stock_training_data(symbol)
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data available for {symbol}")

        # Perform analysis
        signals, used_features = analyzer.analyze_stock(df, selected_features)
        
        return {
            "symbol": symbol,
            "signals": signals,
            "features_used": used_features,
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id
        }
        
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))





@router.get("/models/{model_id}/info")
async def get_model_info(
    model_id: str,
    current_user: str = Depends(get_current_user)
):
    try:
        # Get model info from database
        model_info = models_collection.find_one({
            "id": model_id,
            "user_email": current_user
        })
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        model_path = model_info.get('path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="Model file not found")

        # Load model to get input shape
        model = tf.keras.models.load_model(model_path)
        input_shape = model.input_shape

        # Get required feature count from input shape
        required_features = input_shape[-1] if len(input_shape) >= 3 else 1

        # Define recommended features based on model type
        # You can customize this based on your model's architecture
        recommended_features = [
            # Core price features
            'Close', 'High', 'Low', 'Open', 'Volume',
            
            # Key technical indicators
            'SMA_20', 'EMA_20', 'MACD', 'RSI',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            
            # Volume indicators
            'OBV', 'AD', 'MFI',
            
            # Trend indicators
            'ADX', 'PLUS_DI', 'MINUS_DI',
            
            # Additional indicators
            'ATR', 'STDDEV', 'ROC',
            'STOCH_K', 'STOCH_D',
            'Price_MA5_Ratio', 'Price_MA20_Ratio',
            'BB_Position', 'Trend_Strength'
        ]

        return {
            "id": model_id,
            "name": model_info.get("name", "Unnamed Model"),
            "input_shape": input_shape,
            "required_features": required_features,
            "recommended_features": recommended_features[:required_features],
            "model_type": model_info.get("model_type", "LSTM"),
            "features": model_info.get("features", []),
            "status": model_info.get("status", "inactive"),
            "created_at": model_info.get("uploadDate", datetime.now(timezone.utc))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        # Create a new ObjectId for the model
        model_id = str(ObjectId())
        
        # Ensure the filename has .keras extension
        base_filename = os.path.splitext(model.filename)[0]
        file_path = os.path.join(SETTINGS.MODEL_DIR, f"{base_filename}.keras")
        
        # Make sure the models directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        logger.info(f"Saving model to: {file_path}")
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(model.file, buffer)
            
        logger.info(f"Model saved successfully at: {file_path}")
        
        # Create the model document
        model_doc = {
            "_id": ObjectId(model_id),
            "id": model_id,  # Store both _id and id
            "user_email": current_user,
            "name": f"{base_filename}.keras",
            "path": file_path,
            "uploadDate": datetime.now(timezone.utc),
            "status": "inactive",
            "features": ["price", "volume", "technical_indicators"]
        }
        
        # Insert into database
        models_collection.insert_one(model_doc)
        logger.info(f"Model document created with ID: {model_id}")
        
        return {
            "message": "Model uploaded successfully",
            "model": {
                "id": model_id,
                "name": model_doc["name"],
                "path": model_doc["path"],
                "uploadDate": model_doc["uploadDate"].isoformat(),
                "status": "inactive",
                "features": model_doc["features"]
            }
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/{model_id}/analyze")
async def analyze_stock(
    model_id: str,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    try:
        logger.debug(f"Starting analysis for model_id: {model_id}")
        logger.debug(f"Current user: {current_user}")
        
        # Find the model document
        model_info = None
        
        # Try all possible ways to find the model
        queries = [
            {"id": model_id, "user_email": current_user},
            {"_id": model_id, "user_email": current_user},
            {"path": {"$regex": f".*{model_id}.*"}, "user_email": current_user},
            {"name": "first_model.keras", "user_email": current_user}  # Fallback to find by name
        ]
        
        for query in queries:
            logger.debug(f"Trying to find model with query: {query}")
            model_info = models_collection.find_one(query)
            if model_info:
                logger.debug(f"Found model info: {model_info}")
                break
                
        if not model_info:
            logger.error(f"Model not found in database for ID: {model_id}")
            # List all models for debugging
            all_models = list(models_collection.find({"user_email": current_user}))
            logger.debug(f"Available models: {all_models}")
            raise HTTPException(
                status_code=404,
                detail=f"Model not found with ID: {model_id}"
            )

        # Get the model path
        model_path = model_info.get('path')
        logger.debug(f"Model path from database: {model_path}")
        
        # Check if path exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at path: {model_path}")
            # Try to find the file in the models directory
            models_dir = SETTINGS.MODEL_DIR
            logger.debug(f"Checking models directory: {models_dir}")
            if os.path.exists(models_dir):
                files = os.listdir(models_dir)
                logger.debug(f"Files in models directory: {files}")
            
            # Try alternative path
            alt_path = os.path.join(SETTINGS.MODEL_DIR, "first_model.keras")
            if os.path.exists(alt_path):
                logger.debug(f"Found model at alternative path: {alt_path}")
                model_path = alt_path
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Model file not found"
                )

        # Load the model
        logger.debug(f"Attempting to load model from: {model_path}")
        try:
            model = load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )

        # Get request body and process
        body = await request.json()
        symbol = body["symbol"].upper()
        logger.debug(f"Processing symbol: {symbol}")

        # Get stock data
        df = await get_stock_training_data(symbol, period="1y")
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No data available for symbol {symbol}"
            )

        # Process data and make predictions
        # ... rest of your prediction code ...

        return {
            "symbol": symbol,
            "signals": signals,
            "analysis_date": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e))

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
        # Convert string ID to ObjectId
        model_object_id = ObjectId(model_id)
        
        # Find the model
        model = models_collection.find_one({
            "_id": model_object_id,
            "user_email": current_user
        })
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Delete the file if it exists
        if "path" in model and os.path.exists(model["path"]):
            os.remove(model["path"])

        # Delete from database
        result = models_collection.delete_one({
            "_id": model_object_id,
            "user_email": current_user
        })

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Model not found")

        return {"message": "Model deleted successfully"}

    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid model ID format")
    except Exception as e:
        logger.error(f"Delete model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
