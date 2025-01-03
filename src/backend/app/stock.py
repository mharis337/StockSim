from datetime import datetime, timedelta, timezone
import math
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import pandas as pd
import pytz
import yfinance as yf
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from config import SETTINGS
from database import (
    users_collection,
    transactions_collection,
    portfolio_snapshots_collection,
)
from auth import get_current_user

router = APIRouter(
    prefix="/api",
    tags=["Stock Management"]
)

def decimal_round(amount: Decimal) -> Decimal:
    if not isinstance(amount, Decimal):
        amount = Decimal(str(amount))
    return amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

def process_dataframe(df: pd.DataFrame) -> list:
    df.reset_index(inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])

    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)

    import talib
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    df['EMA_20'] = talib.EMA(close, timeperiod=20)

    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    df['RSI'] = talib.RSI(close, timeperiod=14)

    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd

    df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
    df['ROC'] = talib.ROC(close, timeperiod=10)
    df['OBV'] = talib.OBV(close, volume)
    df['AD'] = talib.AD(high, low, close, volume)
    df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    df['STDDEV'] = talib.STDDEV(close, timeperiod=20, nbdev=1)
    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

    df['TENKAN_SEN'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['KIJUN_SEN'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['SENKOU_SPAN_A'] = ((df['TENKAN_SEN'] + df['KIJUN_SEN']) / 2).shift(26)
    df['SENKOU_SPAN_B'] = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2
    df['SENKOU_SPAN_B'] = df['SENKOU_SPAN_B'].shift(26)
    df['CHIKO_SPAN'] = df['Close'].shift(-26)

    df['PIVOT'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['PIVOT']) - df['Low']
    df['S1'] = (2 * df['PIVOT']) - df['High']

    critical_columns = ['Close', 'High', 'Low', 'Open']
    df.dropna(subset=critical_columns, inplace=True)

    df = df.where(pd.notnull(df), None)

    data_json = df.to_dict(orient="records")

    for record in data_json:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None

    return data_json

def align_to_market_open(df: pd.DataFrame, interval: str, market_open_time: str = "09:30") -> pd.DataFrame:
    if interval not in ["1m", "2m", "5m", "1h"]:
        return df

    market_open_hour, market_open_minute = map(int, market_open_time.split(":"))
    eastern = pytz.timezone("US/Eastern")

    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize('UTC').dt.tz_convert(eastern)
    else:
        df["Date"] = df["Date"].dt.tz_convert(eastern)

    first_timestamp = df["Date"].iloc[0]
    market_open = first_timestamp.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
    offset = market_open - first_timestamp
    df["Date"] = df["Date"] + offset

    return df

@router.post("/transaction")
async def create_transaction(
    transaction: dict,
    current_user: str = Depends(get_current_user)
):
    try:
        user = users_collection.find_one({"email": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        quantity = int(transaction["quantity"])
        price = decimal_round(Decimal(str(transaction["price"])))
        transaction_type = transaction["type"].lower()
        symbol = transaction["symbol"].upper()

        total_cost = Decimal(str(quantity)) * price
        current_balance = Decimal(str(user.get("balance", 0)))

        if transaction_type == "buy":
            if total_cost > current_balance:
                raise HTTPException(status_code=400, detail="Insufficient funds")
            new_balance = decimal_round(current_balance - total_cost)
        elif transaction_type == "sell":
            portfolio = list(transactions_collection.find({
                "user_email": current_user, 
                "symbol": symbol
            }))
            shares_owned = sum(
                tx["quantity"] if tx["type"] == "buy" else -tx["quantity"]
                for tx in portfolio
            )

            if quantity > shares_owned:
                raise HTTPException(status_code=400, detail="Insufficient shares")
            new_balance = decimal_round(current_balance + total_cost)
        else:
            raise HTTPException(status_code=400, detail="Invalid transaction type")

        users_collection.update_one(
            {"email": current_user},
            {"$set": {"balance": float(new_balance)}}
        )

        transaction_record = {
            "user_email": current_user,
            "symbol": symbol,
            "quantity": quantity,
            "price": float(price),
            "total": float(decimal_round(total_cost)),
            "type": transaction_type,
            "timestamp": datetime.now(timezone.utc)
        }
        transactions_collection.insert_one(transaction_record)

        return {
            "message": "Transaction successful",
            "newBalance": float(new_balance),
            "transaction": {
                "symbol": symbol,
                "quantity": quantity,
                "price": float(price),
                "total": float(decimal_round(total_cost)),
                "type": transaction_type
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/balance")
async def get_user_balance(current_user: str = Depends(get_current_user)):
    try:
        user = users_collection.find_one({"email": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"balance": float(user.get("balance", 0))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.options("/user/balance")
async def balance_options():
    return {"message": "OK"}

@router.get("/portfolio")
async def get_portfolio(current_user: str = Depends(get_current_user)):
    try:
        transactions = list(transactions_collection.find({"user_email": current_user}))

        holdings = {}
        for tx in transactions:
            symbol = tx["symbol"]
            quantity = int(tx["quantity"])
            tx_type = tx["type"]

            if symbol not in holdings:
                holdings[symbol] = 0

            if tx_type == "buy":
                holdings[symbol] += quantity
            elif tx_type == "sell":
                holdings[symbol] -= quantity

        portfolio = []
        total_equity = Decimal('0.00')

        for symbol, quantity in holdings.items():
            if quantity > 0:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")

                    if not hist.empty:
                        current_price = decimal_round(Decimal(str(hist['Close'].iloc[-1])))
                        market_value = decimal_round(current_price * quantity)
                        total_equity += market_value

                        portfolio_item = {
                            "symbol": symbol,
                            "quantity": quantity,
                            "current_price": float(current_price),
                            "market_value": float(market_value)
                        }
                        portfolio.append(portfolio_item)
                except Exception:
                    continue

        return {
            "portfolio": portfolio,
            "total_equity": float(total_equity.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio/history")
async def get_portfolio_history(current_user: str = Depends(get_current_user)):
    try:
        user = users_collection.find_one({"email": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        cash_balance = float(user.get("balance", 0))

        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        yesterday = now - timedelta(days=1)
        yesterday = yesterday.replace(hour=16, minute=0, second=0, microsecond=0)

        transactions = list(transactions_collection.find(
            {"user_email": current_user}
        ).sort("timestamp", 1))

        holdings = {}
        total_equity = Decimal('0.00')

        for tx in transactions:
            symbol = tx["symbol"]
            quantity = tx["quantity"]
            if tx["type"] == "buy":
                holdings[symbol] = holdings.get(symbol, 0) + quantity
            else:
                holdings[symbol] = holdings.get(symbol, 0) - quantity

        for symbol, quantity in holdings.items():
            if quantity > 0:
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = Decimal(str(ticker.history(period='1d')['Close'].iloc[-1]))
                    market_value = decimal_round(current_price * quantity)
                    total_equity += market_value
                except Exception:
                    continue

        total_value = Decimal(str(cash_balance)) + total_equity

        realized_pl = Decimal('0.00')
        cost_basis = {}

        for tx in transactions:
            symbol = tx["symbol"]
            quantity = tx["quantity"]
            price = Decimal(str(tx["price"]))

            if tx["type"] == "buy":
                if symbol not in cost_basis:
                    cost_basis[symbol] = {"quantity": 0, "total_cost": Decimal('0.00')}

                current = cost_basis[symbol]
                new_quantity = current["quantity"] + quantity
                new_total_cost = current["total_cost"] + (quantity * price)
                cost_basis[symbol] = {
                    "quantity": new_quantity,
                    "total_cost": new_total_cost,
                    "avg_price": new_total_cost / new_quantity if new_quantity > 0 else Decimal('0.00')
                }
            else:
                if symbol in cost_basis:
                    avg_cost = cost_basis[symbol]["avg_price"]
                    realized_pl += Decimal(str(quantity)) * (price - avg_cost)

                    remaining_quantity = cost_basis[symbol]["quantity"] - quantity
                    if remaining_quantity > 0:
                        cost_basis[symbol]["quantity"] = remaining_quantity
                        cost_basis[symbol]["total_cost"] = remaining_quantity * avg_cost
                    else:
                        cost_basis[symbol]["quantity"] = 0
                        cost_basis[symbol]["total_cost"] = Decimal('0.00')

        unrealized_pl = Decimal('0.00')
        for symbol, position in holdings.items():
            if position > 0 and symbol in cost_basis:
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = Decimal(str(ticker.history(period='1d')['Close'].iloc[-1]))
                    unrealized_pl += Decimal(str(position)) * (current_price - cost_basis[symbol]["avg_price"])
                except Exception:
                    continue

        total_pl = realized_pl + unrealized_pl

        yesterday_snapshot = portfolio_snapshots_collection.find_one({
            "user_email": current_user,
            "timestamp": {
                "$lt": now,
                "$gte": yesterday
            }
        })

        yesterday_value = yesterday_snapshot["total_value"] if yesterday_snapshot else Decimal(str(user.get("initial_balance", 1000)))

        day_pl = total_value - Decimal(str(yesterday_value))
        day_pl_percent = (day_pl / Decimal(str(yesterday_value)) * 100) if yesterday_value > 0 else Decimal('0.00')

        initial_investment = Decimal(str(user.get("initial_balance", 1000)))
        total_pl_percent = (total_pl / initial_investment * 100) if initial_investment > 0 else Decimal('0.00')

        portfolio_snapshots_collection.insert_one({
            "user_email": current_user,
            "timestamp": now,
            "total_value": float(total_value),
            "equity": float(total_equity),
            "cash_balance": cash_balance
        })

        recent_transactions = []
        for tx in reversed(transactions[-10:]):
            tx_dict = {
                "symbol": tx["symbol"],
                "quantity": tx["quantity"],
                "price": tx["price"],
                "type": tx["type"],
                "timestamp": tx["timestamp"]
            }
            recent_transactions.append(tx_dict)

        return {
            "cashBalance": float(cash_balance),
            "equity": float(total_equity),
            "totalValue": float(total_value),
            "dayPL": float(day_pl),
            "dayPLPercent": float(day_pl_percent),
            "totalPL": float(total_pl),
            "totalPLPercent": float(total_pl_percent),
            "realizedPL": float(realized_pl),
            "unrealizedPL": float(unrealized_pl),
            "recentTransactions": recent_transactions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/{symbol}")
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
            "symbol": symbol.upper(),
            "interval": interval,
            "timeframe": timeframe,
            "data": data_json,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historic_stock/{symbol}")
async def get_historic_stock_data(
    symbol: str, 
    interval: str = "1d",  
    timeframe: str = "5y",
    current_user: str = Depends(get_current_user)
):
    try:
        df = yf.download(tickers=symbol, period=timeframe, interval=interval)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        data_json = process_dataframe(df)

        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "timeframe": timeframe,
            "data": data_json,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
