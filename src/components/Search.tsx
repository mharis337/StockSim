"use client";

import React, { useState, useEffect } from "react";
import StockChart from "./StockChart";

const Search = () => {
  const [symbol, setSymbol] = useState("");
  const [timeframe, setTimeframe] = useState("1d");
  const [interval, setIntervalValue] = useState("1h"); // Renamed to avoid conflict with window.setInterval
  const [stockData, setStockData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [autoUpdate, setAutoUpdate] = useState(false);

  const fetchData = async () => {
    try {
      const response = await fetch(
        `http://localhost:5000/api/stock/${symbol}?interval=${interval}&timeframe=${timeframe}`
      );
      const result = await response.json();

      if (response.ok) {
        console.log("Data from backend:", result.data);
        setStockData(result.data);
        setError(null);
      } else {
        console.error("Error:", result.detail);
        setError(result.detail || "Unknown error");
        setStockData([]);
      }
    } catch (err) {
      console.error("Failed to fetch data:", err);
      setError("Failed to fetch stock data");
      setStockData([]);
    }
  };

  // Handle auto-updates for "Now" mode
  useEffect(() => {
    if (autoUpdate && symbol) {
      fetchData(); // Initial fetch
      const intervalId = setInterval(fetchData, 60000); // Fetch every minute
      return () => clearInterval(intervalId); // Cleanup on unmount
    }
  }, [autoUpdate, symbol]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (interval === "1m" && timeframe === "1h") {
      setAutoUpdate(true); // Enable auto-update for "Now"
    } else {
      setAutoUpdate(false); // Disable auto-update
      fetchData();
    }
  };

  return (
    <div style={{ maxWidth: "700px", margin: "100px auto" }}>
      <form className="flex flex-col gap-4" onSubmit={handleSearch}>
        <div className="flex items-center">
          <label htmlFor="symbol" className="sr-only">
            Stock Symbol
          </label>
          <input
            type="text"
            id="symbol"
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
            placeholder="Enter Stock Symbol..."
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            required
          />
        </div>

        <div className="flex items-center gap-4">
          <label htmlFor="timeframe" className="block text-sm font-medium text-gray-700">
            Period (Timeframe):
          </label>
          <select
            id="timeframe"
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm focus:ring-blue-500 focus:border-blue-500 block p-2.5"
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
          >
            <option value="1d">1 Day</option>
            <option value="5d">5 Days</option>
            <option value="1mo">1 Month</option>
            <option value="3mo">3 Months</option>
            <option value="6mo">6 Months</option>
            <option value="1y">1 Year</option>
            <option value="2y">2 Years</option>
            <option value="5y">5 Years</option>
            <option value="10y">10 Years</option>
            <option value="ytd">Year to Date</option>
            <option value="max">Max</option>
          </select>

          <label htmlFor="interval" className="block text-sm font-medium text-gray-700">
            Interval:
          </label>
          <select
            id="interval"
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm focus:ring-blue-500 focus:border-blue-500 block p-2.5"
            value={interval}
            onChange={(e) => setIntervalValue(e.target.value)}
          >
            <option value="1m">1 minute</option>
            <option value="15m">15 minutes</option>
            <option value="30m">30 minutes</option>
            <option value="60m">60 minutes</option>
            <option value="90m">90 minutes</option>
            <option value="1h">1 hour</option>
            <option value="1d">1 day</option>
            <option value="5d">5 days</option>
            <option value="1wk">1 week</option>
            <option value="1mo">1 month</option>
            <option value="3mo">3 months</option>
          </select>
        </div>

        <button
          type="submit"
          className="inline-flex items-center py-2 px-3 text-sm font-medium text-white bg-blue-700 border border-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300"
        >
          Search
        </button>
      </form>

      {error && <p className="text-red-500 mt-4">{error}</p>}

      {stockData.length > 0 && (
        <div className="mt-4">
          <h2 className="text-lg font-semibold text-gray-900">Stock Price Chart:</h2>
          <StockChart data={stockData} interval={interval} />
        </div>
      )}
    </div>
  );
};

export default Search;
