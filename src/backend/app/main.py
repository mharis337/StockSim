from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import pytz

app = FastAPI(title="Stock API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_INTERVALS = [
    "15m", "30m", 
    "60m", "90m", "1h", 
    "1d", "5d", "1wk", "1mo", "3mo"
]

VALID_PERIODS = [
    "1d", "5d", "1wk", "1mo", "3mo", 
    "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
]

TIMEFRAME_MAP = {
    "1d": "1d",
    "5d": "5d",
    "1wk": "1w",
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "10y": "10y",
    "ytd": "ytd",
    "max": "max"
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock API (yfinance version)"}

import pytz

def align_to_market_open(df, interval, market_open_time="09:30"):
    """
    Align timestamps to market open (9:30 AM ET) only for specific intervals (1m, 2m, 5m, 1h).
    For other intervals, the timestamps are left unchanged.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Date' column with tz-aware timestamps.
        interval (str): The interval of the data (e.g., 1m, 2m, 5m, 1h).
        market_open_time (str): Market open time in HH:MM format.

    Returns:
        pd.DataFrame: DataFrame with adjusted timestamps.
    """

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
def get_stock_data(symbol: str, interval: str = "1h", timeframe: str = "5d"):
    """
    Fetch OHLCV data from yfinance for the given symbol, flattened to:
      - Date (ISO string, e.g. 2024-12-27T15:30:00)
      - Open
      - High
      - Low
      - Close
      - Volume
    """
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
