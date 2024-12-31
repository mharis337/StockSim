from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from pymongo import MongoClient
import yfinance as yf
import pandas as pd
from fastapi.responses import JSONResponse
import pytz
import os
from dotenv import load_dotenv

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
    def round_money(amount: float) -> float:
        return round(float(amount), 2)
    
    try:
        user = users_collection.find_one({"email": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get and validate basic transaction data
        quantity = int(transaction["quantity"])
        price = round_money(transaction["price"])
        transaction_type = transaction["type"]
        
        # Calculate total with proper rounding
        total_cost = round_money(price * quantity)
        current_balance = round_money(user.get("balance", 0))

        # Validate the transaction
        if transaction_type == "buy":
            if total_cost > current_balance:
                raise HTTPException(status_code=400, detail="Insufficient funds")
            new_balance = round_money(current_balance - total_cost)
        else:  # sell
            # Verify user has enough shares to sell
            portfolio = list(db["transactions"].find({"user_email": current_user, "symbol": transaction["symbol"]}))
            shares_owned = sum([
                tx["quantity"] if tx["type"] == "buy" else -tx["quantity"]
                for tx in portfolio
            ])
            
            if quantity > shares_owned:
                raise HTTPException(status_code=400, detail="Insufficient shares")
            
            new_balance = round_money(current_balance + total_cost)

        # Update user's balance
        users_collection.update_one(
            {"email": current_user},
            {"$set": {"balance": new_balance}}
        )

        # Record the transaction with exact decimal precision
        transaction_record = {
            "user_email": current_user,
            "symbol": transaction["symbol"],
            "quantity": quantity,
            "price": price,
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
                "price": price,
                "total": total_cost,
                "type": transaction_type
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio(current_user: str = Depends(get_current_user)):
    try:
        def to_decimal(value):
            return round(float(value), 2)

        # Get all transactions for the user
        transactions = list(db["transactions"].find({"user_email": current_user}))

        # Calculate holdings with precise decimal handling
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

        # Prepare portfolio data with exact decimals
        portfolio = []
        total_equity = 0
        
        for symbol, quantity in holdings.items():
            if quantity > 0:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_price = to_decimal(float(hist['Close'].iloc[-1]))
                        market_value = to_decimal(current_price * quantity)
                        total_equity = to_decimal(total_equity + market_value)
                        
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

        return {"portfolio": portfolio, "total_equity": total_equity}
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
