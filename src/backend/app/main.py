# Standard Library Imports
import asyncio
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import traceback
import logging

# Third-Party Imports
import numpy as np
import pandas as pd
import talib
import pytz
import tensorflow as tf
import yfinance as yf
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
import math

# FastAPI Imports
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer

# Pydantic Imports
from pydantic import BaseModel

# TensorFlow Keras Imports
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


# Define base path for models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)


load_dotenv()

# FastAPI app initialization
app = FastAPI(title="Stock API with Auth", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)
logger = logging.getLogger(__name__)

# Add OPTIONS endpoint for model training
@app.options("/api/model/train")
async def model_train_options():
    return {"message": "OK"}
# JWT Configuration
SECRET_KEY = "your-secret-key"  # Change this to a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise RuntimeError("MONGO_URI environment variable not set")

client = MongoClient(mongo_uri)
db = client["StockSim"]
users_collection = db["users"]

# Security utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        return email
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock API with Auth"}

@app.post("/api/register")
async def register(user: User):
    # Check if the user already exists
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Hash the user's password
    hashed_password = pwd_context.hash(user.password)
    
    # Insert the new user into the database with a default balance and empty holdings
    users_collection.insert_one({
        "email": user.email,
        "password": hashed_password,
        "balance": 1000,  # Initial balance
        "holdings": {}     # Explicitly empty holdings
    })
    
    return {"message": "User registered successfully with $1000 balance and 0 stocks"}

@app.post("/api/login", response_model=Token)
async def login(user: User):
    try:
        user_data = users_collection.find_one({"email": user.email})
        
        if not user_data or not pwd_context.verify(user.password, user_data["password"]):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid email or password"}
            )

        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        
        # Create the response
        response = JSONResponse(
            content={
                "access_token": access_token,
                "token_type": "bearer"
            }
        )
        
        # Set cookie
        response.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # 120 minutes
        )
        
        return response
        
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

@app.post("/api/logout")
async def logout():
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie("auth_token")  # Clear the token cookie
    return response

@app.post("/api/refresh")
async def refresh_token(current_user: str = Depends(get_current_user)):
    try:
        # Create a new access token
        access_token = create_access_token(
            data={"sub": current_user},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        # Create the response
        response = JSONResponse(
            content={
                "access_token": access_token,
                "token_type": "bearer"
            }
        )
        
        # Set cookie
        response.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # 120 minutes
        )
        
        return response
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not refresh token"
        )

@app.options("/api/user/balance")
async def balance_options():
    return {"message": "OK"}

@app.get("/api/user/balance")
async def get_user_balance(current_user: str = Depends(get_current_user)):
    try:
        user = users_collection.find_one({"email": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"balance": float(user.get("balance", 0))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transaction")
async def create_transaction(
    transaction: dict,
    current_user: str = Depends(get_current_user)
):
    from decimal import Decimal, ROUND_HALF_UP
    
    def decimal_round(amount: Decimal) -> Decimal:
        """Converts amount to Decimal if it isn't already and rounds"""
        if not isinstance(amount, Decimal):
            amount = Decimal(str(amount))
        return amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    try:
        user = users_collection.find_one({"email": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Convert all monetary values to Decimal immediately
        quantity = int(transaction["quantity"])
        price = decimal_round(Decimal(str(transaction["price"])))
        transaction_type = transaction["type"]
        
        # Calculate total with Decimal arithmetic - NO intermediate rounding
        total_cost = Decimal(str(quantity)) * price
        current_balance = Decimal(str(user.get("balance", 0)))

        # Only round at the final stage when updating balances/saving
        if transaction_type == "buy":
            if total_cost > current_balance:
                raise HTTPException(status_code=400, detail="Insufficient funds")
            new_balance = decimal_round(current_balance - total_cost)
        else:  # sell
            # Verify user has enough shares to sell
            portfolio = list(db["transactions"].find({
                "user_email": current_user, 
                "symbol": transaction["symbol"]
            }))
            shares_owned = sum(
                tx["quantity"] if tx["type"] == "buy" else -tx["quantity"]
                for tx in portfolio
            )
            
            if quantity > shares_owned:
                raise HTTPException(status_code=400, detail="Insufficient shares")
            
            new_balance = decimal_round(current_balance + total_cost)

        # Round everything when saving to database
        total_cost = float(decimal_round(total_cost))
        new_balance = float(decimal_round(new_balance))

        # Update user's balance
        users_collection.update_one(
            {"email": current_user},
            {"$set": {"balance": new_balance}}
        )

        # Record the transaction
        transaction_record = {
            "user_email": current_user,
            "symbol": transaction["symbol"],
            "quantity": quantity,
            "price": float(price),
            "total": total_cost,
            "type": transaction_type,
            "timestamp": datetime.now(timezone.utc)
        }
        db["transactions"].insert_one(transaction_record)
        
        return {
            "message": "Transaction successful",
            "newBalance": new_balance,
            "transaction": {
                "symbol": transaction["symbol"],
                "quantity": quantity,
                "price": float(price),
                "total": total_cost,
                "type": transaction_type
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio(current_user: str = Depends(get_current_user)):
    from decimal import Decimal, ROUND_HALF_UP
    
    def decimal_round(amount: float) -> float:
        return float(Decimal(str(amount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    
    try:
        # Get all transactions for the user
        transactions = list(db["transactions"].find({"user_email": current_user}))

        # Calculate holdings using Decimal for precise arithmetic
        holdings = {}
        for tx in transactions:
            symbol = tx["symbol"]
            quantity = int(tx["quantity"])
            tx_type = tx["type"]
            
            if symbol not in holdings:
                holdings[symbol] = 0
                
            if tx_type.lower() == "buy":
                holdings[symbol] += quantity
            elif tx_type.lower() == "sell":
                holdings[symbol] -= quantity

        # Prepare portfolio data
        portfolio = []
        total_equity = Decimal('0.00')
        
        for symbol, quantity in holdings.items():
            if quantity > 0:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_price = decimal_round(float(hist['Close'].iloc[-1]))
                        market_value = decimal_round(current_price * quantity)
                        total_equity += Decimal(str(market_value))
                        
                        portfolio_item = {
                            "symbol": symbol,
                            "quantity": quantity,
                            "current_price": current_price,
                            "market_value": market_value
                        }
                        portfolio.append(portfolio_item)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    continue

        return {
            "portfolio": portfolio,
            "total_equity": float(total_equity.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/history")
async def get_portfolio_history(current_user: str = Depends(get_current_user)):
    try:
        # Fetch user data
        user = users_collection.find_one({"email": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get current cash balance
        cash_balance = float(user.get("balance", 0))
        
        # Get yesterday's date in Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        yesterday = now - timedelta(days=1)
        yesterday = yesterday.replace(hour=16, minute=0, second=0, microsecond=0)  # 4 PM ET

        # Fetch all transactions
        transactions = list(db["transactions"].find(
            {"user_email": current_user}
        ).sort("timestamp", 1))

        # Calculate current holdings and their market value
        holdings = {}
        total_equity = 0
        
        for tx in transactions:
            symbol = tx["symbol"]
            quantity = tx["quantity"]
            if tx["type"].lower() == "buy":
                holdings[symbol] = holdings.get(symbol, 0) + quantity
            else:  # sell
                holdings[symbol] = holdings.get(symbol, 0) - quantity

        # Calculate current market value of holdings
        for symbol, quantity in holdings.items():
            if quantity > 0:
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    total_equity += quantity * current_price
                except Exception as e:
                    print(f"Error fetching price for {symbol}: {e}")
                    continue

        # Calculate total portfolio value
        total_value = cash_balance + total_equity

        # Calculate realized and unrealized P/L
        realized_pl = 0
        cost_basis = {}  # Keep track of average cost basis per symbol
        
        for tx in transactions:
            symbol = tx["symbol"]
            quantity = tx["quantity"]
            price = tx["price"]
            
            if tx["type"].lower() == "buy":
                if symbol not in cost_basis:
                    cost_basis[symbol] = {"quantity": 0, "total_cost": 0}
                
                # Update cost basis
                current = cost_basis[symbol]
                new_quantity = current["quantity"] + quantity
                new_total_cost = current["total_cost"] + (quantity * price)
                cost_basis[symbol] = {
                    "quantity": new_quantity,
                    "total_cost": new_total_cost,
                    "avg_price": new_total_cost / new_quantity if new_quantity > 0 else 0
                }
            else:  # sell
                if symbol in cost_basis:
                    # Calculate realized P/L for this sale
                    avg_cost = cost_basis[symbol]["avg_price"]
                    realized_pl += quantity * (price - avg_cost)
                    
                    # Update remaining quantity and cost basis
                    remaining_quantity = cost_basis[symbol]["quantity"] - quantity
                    if remaining_quantity > 0:
                        cost_basis[symbol]["quantity"] = remaining_quantity
                        cost_basis[symbol]["total_cost"] = remaining_quantity * avg_cost
                    else:
                        cost_basis[symbol]["quantity"] = 0
                        cost_basis[symbol]["total_cost"] = 0

        # Calculate unrealized P/L
        unrealized_pl = 0
        for symbol, position in holdings.items():
            if position > 0 and symbol in cost_basis:
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    unrealized_pl += position * (current_price - cost_basis[symbol]["avg_price"])
                except Exception:
                    continue

        # Calculate total P/L
        total_pl = realized_pl + unrealized_pl

        # Get yesterday's portfolio value for day P/L calculation
        yesterday_value = 0
        yesterday_snapshot = db["portfolio_snapshots"].find_one({
            "user_email": current_user,
            "timestamp": {
                "$lt": now,
                "$gte": yesterday
            }
        })

        if yesterday_snapshot:
            yesterday_value = yesterday_snapshot["total_value"]
        else:
            # If no snapshot exists, consider initial investment as base
            initial_investment = user.get("initial_balance", 1000)  # Default starting balance
            yesterday_value = initial_investment

        # Calculate day P/L
        day_pl = total_value - yesterday_value
        day_pl_percent = (day_pl / yesterday_value * 100) if yesterday_value > 0 else 0

        # Calculate total P/L percentage
        initial_investment = user.get("initial_balance", 1000)  # Default starting balance
        total_pl_percent = (total_pl / initial_investment * 100) if initial_investment > 0 else 0

        # Store today's snapshot
        db["portfolio_snapshots"].insert_one({
            "user_email": current_user,
            "timestamp": now,
            "total_value": total_value,
            "equity": total_equity,
            "cash_balance": cash_balance
        })

        # Format transactions for response
        recent_transactions = []
        for tx in reversed(transactions[-10:]):  # Last 10 transactions
            tx_dict = {
                "symbol": tx["symbol"],
                "quantity": tx["quantity"],
                "price": tx["price"],
                "type": tx["type"],
                "timestamp": tx["timestamp"]
            }
            recent_transactions.append(tx_dict)

        return {
            "cashBalance": cash_balance,
            "equity": total_equity,
            "totalValue": total_value,
            "dayPL": day_pl,
            "dayPLPercent": day_pl_percent,
            "totalPL": total_pl,
            "totalPLPercent": total_pl_percent,
            "realizedPL": realized_pl,
            "unrealizedPL": unrealized_pl,
            "recentTransactions": recent_transactions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    return {"message": "You are authorized", "user": current_user}

def align_to_market_open(df, interval, market_open_time="09:30"):
    if interval not in ["1m", "2m", "5m", "1h"]:
        return df

    market_open_hour, market_open_minute = map(int, market_open_time.split(":"))

    eastern = pytz.timezone("US/Eastern")
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_convert(eastern)

    first_timestamp = df["Date"].iloc[0]

    market_open = first_timestamp.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)

    offset = market_open - first_timestamp

    df["Date"] = df["Date"] + offset

    return df

@app.get("/api/stock/{symbol}")
async def get_stock_data(
    symbol: str, 
    interval: str = "1h", 
    timeframe: str = "5d",
    current_user: str = Depends(get_current_user)
):
    try:
        df = yf.download(tickers=symbol, period=timeframe, interval=interval)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        df.reset_index(inplace=True)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])

        df = align_to_market_open(df, interval)

        if "Adj Close" in df.columns:
            df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)

        data_json = df.to_dict(orient="records")

        return {
            "symbol": symbol,
            "interval": interval,
            "timeframe": timeframe,
            "data": data_json,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_dataframe(df: pd.DataFrame) -> list:
    """
    Process the DataFrame by calculating technical indicators,
    replacing NaNs with None, and converting to a list of records.
    """
    # Reset index to ensure 'Date' is a column
    df.reset_index(inplace=True)

    # Ensure 'Date' column is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])

    # Align to market open times
    #df = align_to_market_open(df, interval)

    # Rename 'Adj Close' to 'AdjClose' if present
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)

    # Calculate Technical Indicators using TA-Lib
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Moving Averages
    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    df['EMA_20'] = talib.EMA(close, timeperiod=20)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # Parabolic SAR
    df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    # RSI
    df['RSI'] = talib.RSI(close, timeperiod=14)

    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd

    # Williams %R
    df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

    # Rate of Change (ROC)
    df['ROC'] = talib.ROC(close, timeperiod=10)

    # On-Balance Volume (OBV)
    df['OBV'] = talib.OBV(close, volume)

    # Chaikin AD
    df['AD'] = talib.AD(high, low, close, volume)

    # Money Flow Index (MFI)
    df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)

    # Average True Range (ATR)
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)

    # Standard Deviation
    df['STDDEV'] = talib.STDDEV(close, timeperiod=20, nbdev=1)

    # Average Directional Movement Index (ADX)
    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

    # Plus Directional Indicator (+DI)
    df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)

    # Minus Directional Indicator (-DI)
    df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

    # Ichimoku Components
    df['TENKAN_SEN'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['KIJUN_SEN'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['SENKOU_SPAN_A'] = ((df['TENKAN_SEN'] + df['KIJUN_SEN']) / 2).shift(26)
    df['SENKOU_SPAN_B'] = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2
    df['SENKOU_SPAN_B'] = df['SENKOU_SPAN_B'].shift(26)
    df['CHIKO_SPAN'] = df['Close'].shift(-26)

    # Pivot Points (Simple Pivot)
    df['PIVOT'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['PIVOT']) - df['Low']
    df['S1'] = (2 * df['PIVOT']) - df['High']

    # Drop rows where critical columns are NaN
    critical_columns = ['Close', 'High', 'Low', 'Open']
    df.dropna(subset=critical_columns, inplace=True)

    # Replace all NaNs with None for JSON serialization
    df = df.where(pd.notnull(df), None)

    # Convert DataFrame to list of dictionaries
    data_json = df.to_dict(orient="records")

    # Final check to replace any remaining NaNs
    for record in data_json:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None

    return data_json

@app.get("/api/historic_stock/{symbol}")
async def get_historic_stock_data(
    symbol: str, 
    interval: str = "1d",  
    timeframe: str = "5y",
    current_user: str = Depends(get_current_user)
):
    try:
        df = yf.download(tickers=symbol, period=timeframe, interval=interval)
        if df.empty:
            logger.warning(f"No data found for symbol: {symbol}")
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")
            logger.info("Dropped MultiIndex from columns.")

        data_json = process_dataframe(df)

        return {
            "symbol": symbol,
            "interval": interval,
            "timeframe": timeframe,
            "data": data_json,
        }

    except Exception as e:
        logger.error(f"Error in get_historic_stock_data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))





### MODEL ASPECT ####

async def get_stock_training_data(symbol: str, period: str = "2y"):
    """Fetch and preprocess stock data for training"""
    try:
        # Fetch historical data
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

@app.post("/api/model/train")
async def train_model(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    try:
        body = await request.json()
        symbol = body["symbol"]
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
            "created_at": datetime.now(),
            "config": config,
            "status": "training"
        }
        db["models"].insert_one(model_info)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Save model
        model_path = os.path.join(MODEL_PATH, model_id)
        model.save(model_path)
        
        # Calculate accuracy metrics
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        accuracy = 1 - np.mean(np.abs((y_test_actual - predictions) / y_test_actual))
        
        # Update model info
        db["models"].update_one(
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/training-status/{model_id}")
async def model_training_status(
    model_id: str,
    current_user: str = Depends(get_current_user)
):
    async def event_generator():
        try:
            model_info = db["models"].find_one({"id": model_id})
            if not model_info:
                yield json.dumps({"error": "Model not found"})
                return
                
            while True:
                model_info = db["models"].find_one({"id": model_id})
                yield json.dumps({
                    "status": model_info["status"],
                    "accuracy": model_info.get("accuracy"),
                    "test_loss": model_info.get("test_loss")
                })
                
                if model_info["status"] == "completed":
                    break
                    
                await asyncio.sleep(1)
                
        except Exception as e:
            yield json.dumps({"error": str(e)})
    
    return EventSourceResponse(event_generator())


@app.post("/api/model/predict")
async def predict(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    try:
        body = await request.json()
        model_id = body["model_id"]
        symbol = body["symbol"]
        
        # Load model and make prediction
        model_path = os.path.join(MODEL_PATH, model_id)
        model = load_model(model_path)
        
        # Fetch recent data
        df = await get_stock_training_data(symbol, period="60d")
        
        # Prepare features (same as training)
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
        sequence = scaled_data[-60:].reshape(1, 60, features.shape[1])
        
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
        raise HTTPException(status_code=500, detail=str(e))






@app.get("/api/models")
async def get_models(current_user: str = Depends(get_current_user)):
    try:
        models = list(db["models"].find({"user_email": current_user}))
        
        # Process models to ensure IDs are properly formatted
        for model in models:
            # Convert ObjectId to string
            model["_id"] = str(model["_id"])
            # Ensure there's also an 'id' field
            model["id"] = model["_id"]
            
        return {"models": models}
    except Exception as e:
        print(f"Error fetching models: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))
    
# Update the model upload endpoint
@app.post("/api/models/upload")
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
        file_path = os.path.join(MODEL_PATH, unique_filename)

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
        result = db["models"].insert_one(model_doc)
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_id}/toggle")
async def toggle_model(
    model_id: str,
    current_user: str = Depends(get_current_user)
):
    print(f"Attempting to toggle model with ID: {model_id}")  # Debug log
    try:
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
            
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(model_id)
        except InvalidId:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model ID format: {model_id}"
            )

        # Find the model and verify ownership
        model = db["models"].find_one({
            "_id": object_id,
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
        result = db["models"].update_one(
            {"_id": object_id},
            {"$set": {"status": new_status}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to update model status"
            )

        return {
            "status": new_status,
            "model_id": str(object_id)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error toggling model: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

# Update the model deletion endpoint
@app.delete("/api/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: str = Depends(get_current_user)
):
    try:
        # Convert string ID to ObjectId
        object_id = ObjectId(model_id)
        
        # Find the model
        model = db["models"].find_one({
            "_id": object_id,
            "user_email": current_user
        })
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Delete the file if it exists
        if os.path.exists(model["path"]):
            os.remove(model["path"])

        # Delete from database
        db["models"].delete_one({"_id": object_id})

        return {"message": "Model deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
