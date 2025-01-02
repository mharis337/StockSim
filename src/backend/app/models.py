from datetime import datetime, timedelta, timezone
import os
import shutil
import json
import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from bson import ObjectId
from bson.errors import InvalidId
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, TFSMLayer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import talib
from sse_starlette.sse import EventSourceResponse

from model_analyzer import ModelAnalyzer
from model_cache import ModelCacheManager, model_cache

from config import SETTINGS
from database import models_collection
from auth import get_current_user

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if not os.path.exists(SETTINGS.MODEL_DIR):
    os.makedirs(SETTINGS.MODEL_DIR, exist_ok=True)
    logger.info(f"Created models directory at: {SETTINGS.MODEL_DIR}")
else:
    logger.info(f"Using existing models directory at: {SETTINGS.MODEL_DIR}")

router = APIRouter(
    prefix="/api",
    tags=["Model Management"]
)

async def get_stock_training_data(symbol: str, period: str = "1y"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        df = df.astype('float64')
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        try:
            df['MA5'] = talib.SMA(close, timeperiod=5)
            df['MA20'] = talib.SMA(close, timeperiod=20)
            df['EMA20'] = talib.EMA(close, timeperiod=20)
            df['RSI'] = talib.RSI(close)
            df['ROC'] = talib.ROC(close, timeperiod=10)
            df['ATR'] = talib.ATR(high, low, close)
            df['BBANDS_U'], df['BBANDS_M'], df['BBANDS_L'] = talib.BBANDS(close)
            df['OBV'] = talib.OBV(close, volume)
            df['AD'] = talib.AD(high, low, close, volume)
            df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
            df['ADX'] = talib.ADX(high, low, close)
            df['PLUS_DI'] = talib.PLUS_DI(high, low, close)
            df['MINUS_DI'] = talib.MINUS_DI(high, low, close)
            df['MACD'], df['MACD_SIGNAL'], _ = talib.MACD(close)
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise ValueError(f"Error calculating indicators: {str(e)}")
        
        df = df.dropna()
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

def prepare_sequences(data, sequence_length=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

async def event_generator(model_id: str):
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
    predicted_change = (predicted_price - current_price) / current_price
    confidence = min(abs(predicted_change) * 200, 100)
    if predicted_change > 0:
        target_price = predicted_price
        stop_loss = current_price * 0.98
    else:
        target_price = predicted_price
        stop_loss = current_price * 1.02
    return target_price, stop_loss, confidence

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    feature_columns = [
        'Close', 'Volume', 
        'MA5', 'MA20', 'EMA20',
        'RSI', 'ROC',
        'ATR', 'BBANDS_U', 'BBANDS_M', 'BBANDS_L',
        'OBV', 'AD', 'MFI',
        'ADX', 'PLUS_DI', 'MINUS_DI',
        'MACD', 'MACD_SIGNAL'
    ]
    for col in feature_columns:
        df[col] = df[col].astype('float64')
    return df[feature_columns].values

@router.post("/model/{model_id}/analyze")
async def analyze_stock(model_id: str, request: Request, current_user: str = Depends(get_current_user)):
    try:
        body = await request.json()
        symbol = body["symbol"].upper()
        selected_features = body.get("features", [])
        
        logger.info(f"Analyzing {symbol} with features: {selected_features}")
        
        model_info = models_collection.find_one({
            "id": model_id,
            "user_email": current_user
        })
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        model_path = model_info.get('path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="Model file not found")

        analyzer = ModelAnalyzer(model_path)
        
        df = await get_stock_training_data(symbol)
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data available for {symbol}")

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
async def get_model_info(model_id: str, current_user: str = Depends(get_current_user)):
    try:
        model_info = models_collection.find_one({
            "id": model_id,
            "user_email": current_user
        })
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        model_path = model_info.get('path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="Model file not found")

        model = load_model(model_path)
        input_shape = model.input_shape
        required_features = input_shape[-1] if len(input_shape) >= 3 else 1

        recommended_features = [
            'Close', 'High', 'Low', 'Open', 'Volume',
            'SMA_20', 'EMA_20', 'MACD', 'RSI',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'OBV', 'AD', 'MFI',
            'ADX', 'PLUS_DI', 'MINUS_DI',
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
async def train_model(request: Request, current_user: str = Depends(get_current_user)):
    try:
        body = await request.json()
        symbol = body["symbol"].upper()
        config = body.get("config", {
            "epochs": 50,
            "batch_size": 32,
            "sequence_length": 60,
            "learning_rate": 0.001
        })
        
        df = await get_stock_training_data(symbol)
        
        features = np.column_stack((
            df['Close'],
            df['Volume'],
            df['MA5'],
            df['MA20'],
            df['RSI'],
            df['MACD']
        ))
        
        X, y, scaler = prepare_sequences(features, config['sequence_length'])
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
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
        
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        model_path = os.path.join(SETTINGS.MODEL_DIR, model_id)
        model.save(model_path)
        
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        accuracy = 1 - np.mean(np.abs((y_test_actual - predictions) / y_test_actual))
        
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
async def model_training_status(model_id: str, current_user: str = Depends(get_current_user)):
    return EventSourceResponse(event_generator(model_id))

@router.post("/model/predict")
async def predict(request: Request, current_user: str = Depends(get_current_user)):
    try:
        body = await request.json()
        model_id = body["model_id"]
        symbol = body["symbol"].upper()
        
        model_path = os.path.join(SETTINGS.MODEL_DIR, model_id)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = load_model(model_path)
        
        df = await get_stock_training_data(symbol, period="60d")
        
        features = np.column_stack((
            df['Close'],
            df['Volume'],
            df['MA5'],
            df['MA20'],
            df['RSI'],
            df['MACD']
        ))
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features)
        sequence_length = int(body.get("sequence_length", 60))
        sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, features.shape[1])
        
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

@router.post("/model/{model_id}/backtest")
async def backtest_model(
    model_id: str, 
    request: Request,
    current_user: str = Depends(get_current_user)
):
    try:
        body = await request.json()
        symbol = body["symbol"].upper()
        lookback_days = body.get("lookback_days", 30)
        
        logger.info(f"Running backtest for {symbol} with lookback of {lookback_days} days")
        
        model_info = models_collection.find_one({
            "id": model_id,
            "user_email": current_user
        })
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        model_path = model_info.get('path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="Model file not found")

        df = await get_stock_training_data(symbol, period="1y")
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data available for {symbol}")

        analyzer = ModelAnalyzer(model_path)
        backtest_results = analyzer.backtest_prediction_accuracy(df, lookback_days)
        
        models_collection.update_one(
            {"id": model_id},
            {
                "$set": {
                    "last_backtest": {
                        "date": datetime.now(timezone.utc),
                        "symbol": symbol,
                        "lookback_days": lookback_days,
                        "results": backtest_results
                    }
                }
            }
        )
        
        return {
            "model_id": model_id,
            "symbol": symbol,
            "backtest_period": backtest_results["lookback_period"],
            "metrics": {
                "direction_accuracy": backtest_results["direction_accuracy"],
                "average_error": backtest_results.get("avg_error", 0),
                "profitability": backtest_results["profitability"],
                "total_trades": backtest_results["total_trades"],
                "profitable_trades": backtest_results["profitable_trades"]
            },
            "features_used": backtest_results["features_used"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_models(current_user: str = Depends(get_current_user)):
    try:
        models = list(models_collection.find({"user_email": current_user}))
        
        for model in models:
            if "_id" in model:
                model["id"] = str(model["_id"])
                del model["_id"]
            
            if "last_backtest" in model:
                backtest = model["last_backtest"]
                model["backtest_summary"] = {
                    "last_tested": backtest["date"].isoformat(),
                    "symbol": backtest["symbol"],
                    "accuracy": backtest["results"]["direction_accuracy"],
                    "profitability": backtest["results"]["profitability"]
                }
        
        return {"models": models}
    except Exception as e:
        logger.error(f"Get models error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/upload")
async def upload_model(model: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        model_id = str(ObjectId())
        base_filename = os.path.splitext(model.filename)[0]
        file_path = os.path.join(SETTINGS.MODEL_DIR, f"{base_filename}.keras")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logger.info(f"Saving model to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(model.file, buffer)
        logger.info(f"Model saved successfully at: {file_path}")
        
        model_doc = {
            "_id": ObjectId(model_id),
            "id": model_id,
            "user_email": current_user,
            "name": f"{base_filename}.keras",
            "path": file_path,
            "uploadDate": datetime.now(timezone.utc),
            "status": "inactive",
            "features": ["price", "volume", "technical_indicators"]
        }
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

@router.post("/models/{model_id}/toggle")
async def toggle_model(model_id: str, current_user: str = Depends(get_current_user)):
    try:
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        model = models_collection.find_one({
            "id": model_id,
            "user_email": current_user
        })
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found with ID: {model_id}")
        current_status = model.get("status", "inactive")
        new_status = "active" if current_status != "active" else "inactive"
        result = models_collection.update_one(
            {"id": model_id},
            {"$set": {"status": new_status}}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update model status")
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
async def delete_model(model_id: str, current_user: str = Depends(get_current_user)):
    try:
        model_object_id = ObjectId(model_id)
        model = models_collection.find_one({
            "_id": model_object_id,
            "user_email": current_user
        })
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        if "path" in model and os.path.exists(model["path"]):
            os.remove(model["path"])
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
