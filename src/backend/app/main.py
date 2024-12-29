from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

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

        print("Before flattening:\n", df.columns)
        print(df.head())


        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        df.reset_index(inplace=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

        if "Adj Close" in df.columns:
            df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)

        data_json = df.to_dict(orient="records")

        print("After flattening:\n", df.columns)
        print(df.head())

        return {
            "symbol": symbol,
            "interval": interval,
            "timeframe": timeframe,
            "data": data_json,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))