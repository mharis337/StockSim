"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import StockChart from "./StockChart";
import StockInfoPanel from "./StockInfoPanel";
import BuySell from "./BuySell";

const Search = () => {
  const router = useRouter();
  const [symbol, setSymbol] = useState("");
  const [timeframe, setTimeframe] = useState("1d");
  const [interval, setIntervalValue] = useState("1h");
  const [stockData, setStockData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [autoUpdate, setAutoUpdate] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const fetchData = useCallback(async () => {
    if (!symbol) return;
    try {
      setIsLoading(true);
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("No authentication token found");
      }
      const response = await fetch(
        `http://localhost:5000/api/stock/${symbol}?interval=${interval}&timeframe=${timeframe}`,
        {
          credentials: "include",
          headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json"
          }
        }
      );
      if (response.status === 401) {
        localStorage.removeItem("token");
        router.push("/login");
        return;
      }
      const result = await response.json();
      if (response.ok) {
        setStockData(result.data);
        setError(null);
      } else {
        setError(result.detail || "Unknown error");
        setStockData([]);
      }
    } catch (err) {
      setError("Failed to fetch stock data");
      setStockData([]);
      if (err instanceof Error && err.message === "No authentication token found") {
        router.push("/login");
      }
    } finally {
      setIsLoading(false);
    }
  }, [symbol, interval, timeframe, router]);

  useEffect(() => {
    if (symbol && autoUpdate) {
      fetchData();
      const intervalId = setInterval(fetchData, 30000);
      return () => clearInterval(intervalId);
    }
  }, [symbol, autoUpdate, fetchData]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setAutoUpdate(true);
    fetchData();
  };

  return (
    <div className="bg-white/50 backdrop-blur-sm rounded-lg shadow-lg"
    style={{ 
      padding:32,
      marginLeft: 32,
      marginRight: 32,
      marginTop: 32,
    }}>
      <form className="flex flex-col gap-4 mb-6" onSubmit={handleSearch}>
        <div className="flex items-center">
          <input
            type="text"
            className="flex-1 p-2 border rounded text-gray-900 text-base font-medium border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            placeholder="Enter Stock Symbol..."
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            required
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-base font-semibold text-gray-900 mb-1">
              Period:
            </label>
            <select
              className="w-full p-2 border rounded text-gray-900 text-base font-medium border-gray-300 bg-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
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
            </select>
          </div>

          <div>
            <label className="block text-base font-semibold text-gray-900 mb-1">
              Interval:
            </label>
            <select
              className="w-full p-2 border rounded text-gray-900 text-base font-medium border-gray-300 bg-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              value={interval}
              onChange={(e) => setIntervalValue(e.target.value)}
            >
              <option value="1m">1 minute</option>
              <option value="5m">5 minutes</option>
              <option value="15m">15 minutes</option>
              <option value="30m">30 minutes</option>
              <option value="60m">60 minutes</option>
              <option value="1h">1 hour</option>
              <option value="1d">1 day</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 transition-colors text-base font-semibold shadow-sm"
          disabled={isLoading}
        >
          {isLoading ? "Loading..." : "Search"}
        </button>
      </form>

      {error && (
        <div className="mb-6 p-3 bg-red-100 border border-red-400 text-red-800 rounded-md font-medium">
          {error}
        </div>
      )}

      {stockData.length > 0 && (
        <>
          <StockInfoPanel data={stockData} symbol={symbol} />
          <div className="mt-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-900">Price Chart</h2>
              {autoUpdate && (
                <div className="text-sm text-green-600 font-medium">
                  Auto-updating every 30 seconds
                </div>
              )}
            </div>
            <StockChart data={stockData} interval={interval} />
          </div>

          <div className="mt-8 border-t pt-8">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Trade {symbol}</h2>
            <BuySell stockData={stockData} symbol={symbol} />
          </div>
        </>
      )}
    </div>
  );
};

export default Search;
