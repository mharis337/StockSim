from datetime import datetime, timedelta
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
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    print("\n=== Auth Debug ===")
    print(f"Received token: {token[:20]}...")  # Only print first 20 chars for security
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        print(f"Decoded email from token: {email}")
        
        if email is None:
            print("No email in token")
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        return email
        
    except JWTError as e:
        print(f"JWT Error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        print(f"Other error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")
    finally:
        print("=== End Auth Debug ===\n")

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
    
    # Insert the new user into the database with a default balance
    users_collection.insert_one({
        "email": user.email,
        "password": hashed_password,
        "balance": 1000  # Initial balance
    })
    return {"message": "User registered successfully with $1000 balance"}

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
            max_age=1800  # 30 minutes
        )
        
        return response
        
    except Exception as e:
        print(f"Login error: {str(e)}")
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
            max_age=1800  # 30 minutes
        )
        
        return response
    except Exception as e:
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
        print(f"Error in get_user_balance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transaction")
async def create_transaction(
    transaction: dict,
    current_user: str = Depends(get_current_user)
):
    user = users_collection.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    current_balance = user.get("balance", 0)
    total_cost = transaction["quantity"] * transaction["price"]

    if transaction["type"] == "buy":
        if total_cost > current_balance:
            raise HTTPException(status_code=400, detail="Insufficient funds")
        new_balance = current_balance - total_cost
    else:  # sell
        new_balance = current_balance + total_cost

    # Update user's balance
    users_collection.update_one(
        {"email": current_user},
        {"$set": {"balance": new_balance}}
    )

    # Record the transaction
    transaction_record = {
        "user_email": current_user,
        "symbol": transaction["symbol"],
        "quantity": transaction["quantity"],
        "price": transaction["price"],
        "type": transaction["type"],
        "timestamp": datetime.utcnow()
    }
    db["transactions"].insert_one(transaction_record)
    
    return {"message": "Transaction successful", "newBalance": new_balance}   


@app.get("/api/portfolio")
async def get_portfolio(current_user: str = Depends(get_current_user)):
    try:
        # Get all transactions for the user
        transactions = list(db["transactions"].find({"user_email": current_user}))
        
        # Calculate holdings
        holdings = {}
        for tx in transactions:
            symbol = tx["symbol"]
            quantity = tx["quantity"] if tx["type"] == "buy" else -tx["quantity"]
            
            if symbol not in holdings:
                holdings[symbol] = 0
            holdings[symbol] += quantity
        
        # Remove symbols with zero quantity
        holdings = {k: v for k, v in holdings.items() if v > 0}
        
        # Get current prices for all held stocks
        portfolio = []
        for symbol, quantity in holdings.items():
            try:
                # Fetch latest price
                stock_data = yf.download(symbol, period="1d", interval="1m")
                if not stock_data.empty:
                    current_price = stock_data['Close'][-1]
                    market_value = current_price * quantity
                    
                    portfolio.append({
                        "symbol": symbol,
                        "quantity": quantity,
                        "current_price": current_price,
                        "market_value": market_value
                    })
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return {"portfolio": portfolio}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/history")
async def get_portfolio_history(current_user: str = Depends(get_current_user)):
    try:
        transactions = list(db["transactions"].find(
            {"user_email": current_user},
            {'_id': 0}  # Exclude MongoDB _id
        ))
        return {"transactions": transactions}
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