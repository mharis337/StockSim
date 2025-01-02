import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StockSidePanelProps {
  symbol: string;
  stockData: any[];
}

export const StockSidePanel: React.FC<StockSidePanelProps> = ({ symbol, stockData }) => {
  if (!stockData.length) return null;

  const latestData = stockData[stockData.length - 1];
  const previousData = stockData[stockData.length - 2];

  const priceChange = latestData.Close - previousData.Close;
  const percentChange = (priceChange / previousData.Close) * 100;

  return (
    <div className="w-64 bg-white p-4 rounded-lg shadow">
      <h3 className="text-xl font-bold mb-4">{symbol.toUpperCase()}</h3>

      <div className="space-y-4">
        <div>
          <div className="text-sm text-gray-600">Price</div>
          <div className="text-2xl font-bold">${latestData.Close.toFixed(2)}</div>
          <div className={`flex items-center ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {priceChange >= 0 ? (
              <TrendingUp className="w-4 h-4 mr-1" />
            ) : (
              <TrendingDown className="w-4 h-4 mr-1" />
            )}
            <span>
              {priceChange.toFixed(2)} ({percentChange.toFixed(2)}%)
            </span>
          </div>
        </div>

        <div className="border-t pt-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-gray-600">Open</div>
              <div className="font-medium">${latestData.Open.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">High</div>
              <div className="font-medium">${latestData.High.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Low</div>
              <div className="font-medium">${latestData.Low.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Volume</div>
              <div className="font-medium">{latestData.Volume.toLocaleString()}</div>
            </div>
          </div>
        </div>

        <div className="border-t pt-4">
          <h4 className="font-medium mb-2">Technical Indicators</h4>
          <div className="space-y-2">
            <div>
              <div className="text-sm text-gray-600">RSI</div>
              <div className="font-medium">{latestData.RSI?.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">MACD</div>
              <div className="font-medium">{latestData.MACD?.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">ADX</div>
              <div className="font-medium">{latestData.ADX?.toFixed(2)}</div>
            </div>
          </div>
        </div>

        <div className="border-t pt-4">
          <h4 className="font-medium mb-2">ML Signals</h4>
          <div className="space-y-2">
            {latestData.buySignal && (
              <div className="bg-green-100 text-green-700 p-2 rounded">
                Buy Signal Detected
              </div>
            )}
            {latestData.sellSignal && (
              <div className="bg-red-100 text-red-700 p-2 rounded">
                Sell Signal Detected
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};