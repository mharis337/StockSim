import React, { useState, useEffect } from 'react';
import { PlusCircle, X, RefreshCw, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface WatchlistStock {
  symbol: string;
  currentPrice: number;
  previousClose: number;
  change: number;
  changePercent: number;
}

const Watchlist = () => {
  const [stocks, setStocks] = useState<WatchlistStock[]>([]);
  const [newSymbol, setNewSymbol] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchStockData = async (symbol: string): Promise<WatchlistStock | null> => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(
        `http://localhost:5000/api/stock/${symbol}?interval=1d&timeframe=5d`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch data for ${symbol}`);
      }

      const data = await response.json();
      if (!data.data || data.data.length < 2) {
        throw new Error(`Insufficient data for ${symbol}`);
      }

      const validData = data.data.filter(d => d && d.Close !== null && d.Close !== undefined);
      if (validData.length < 2) {
        throw new Error(`Insufficient valid data for ${symbol}`);
      }

      const latestData = validData[validData.length - 1];
      const previousData = validData[validData.length - 2];

      const change = latestData.Close - previousData.Close;
      const changePercent = (change / previousData.Close) * 100;

      return {
        symbol: symbol.toUpperCase(),
        currentPrice: latestData.Close,
        previousClose: previousData.Close,
        change,
        changePercent
      };
    } catch (error) {
      setError(error instanceof Error ? error.message : `Failed to fetch data for ${symbol}`);
      return null;
    }
  };

  const refreshWatchlist = async () => {
    setIsRefreshing(true);
    setError(null);
    try {
      const updatedStocks = await Promise.all(
        stocks.map(stock => fetchStockData(stock.symbol))
      );
      setStocks(updatedStocks.filter((stock): stock is WatchlistStock => stock !== null));
    } catch (error) {
      setError('Failed to refresh watchlist');
    } finally {
      setIsRefreshing(false);
    }
  };

  const addStock = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSymbol) return;

    setIsLoading(true);
    setError(null);

    try {
      const stockData = await fetchStockData(newSymbol);
      if (stockData) {
        setStocks(prev => [...prev, stockData]);
        setNewSymbol('');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const removeStock = (symbolToRemove: string) => {
    setStocks(prev => prev.filter(stock => stock.symbol !== symbolToRemove));
  };

  useEffect(() => {
    const intervalId = setInterval(refreshWatchlist, 60000);
    return () => clearInterval(intervalId);
  }, [stocks]);

  return (
    <div className="space-y-6 p-6">
      <div className="bg-white/50 backdrop-blur-sm rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">Watchlist</h2>
          <button
            onClick={refreshWatchlist}
            disabled={isRefreshing}
            className="text-gray-500 hover:text-gray-700 transition-colors"
          >
            <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>
        </div>

        <form onSubmit={addStock} className="mb-6">
          <div className="flex gap-2">
            <input
              type="text"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
              placeholder="Enter stock symbol..."
              className="flex-1 p-2 border border-gray-200 rounded-md bg-white/50 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <PlusCircle className="w-5 h-5" />
            </button>
          </div>
        </form>

        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="space-y-4">
          {stocks.map((stock) => (
            <div
              key={stock.symbol}
              className="flex items-center justify-between p-4 border border-gray-200 rounded-lg bg-white/30 hover:bg-white/40 transition-colors"
            >
              <div className="flex-1">
                <div className="flex justify-between items-center">
                  <span className="text-lg font-semibold text-gray-800">
                    {stock.symbol}
                  </span>
                  <button
                    onClick={() => removeStock(stock.symbol)}
                    className="text-gray-400 hover:text-red-500 transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                <div className="flex items-center space-x-4 mt-2">
                  <span className="text-xl font-bold text-gray-800">
                    ${stock.currentPrice.toFixed(2)}
                  </span>
                  <span
                    className={`text-sm font-semibold ${
                      stock.change >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {stock.change >= 0 ? '+' : ''}
                    {stock.change.toFixed(2)} ({stock.changePercent.toFixed(2)}%)
                  </span>
                </div>
              </div>
            </div>
          ))}

          {stocks.length === 0 && !error && (
            <div className="text-center py-8 text-gray-500">
              No stocks in watchlist. Add some stocks to track them.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Watchlist;
