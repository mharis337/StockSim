import React from 'react';

interface StockInfoPanelProps {
  data: any[];
  symbol: string;
}

const StockInfoPanel = ({ data, symbol }: StockInfoPanelProps) => {
  if (!data || data.length === 0) return null;

  const currentData = data[data.length - 1];
  const previousData = data[data.length - 2];

  const priceChange = currentData.Close - previousData.Close;
  const percentChange = (priceChange / previousData.Close) * 100;
  const dayHigh = Math.max(...data.map(d => d.High));
  const dayLow = Math.min(...data.map(d => d.Low));
  const volume = data.reduce((sum, d) => sum + d.Volume, 0);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6 p-6 bg-white rounded-lg border border-gray-200 shadow-sm">
      {/* Symbol and Current Price */}
      <div className="flex flex-col">
        <div className="text-2xl font-bold text-gray-800 mb-1">{symbol}</div>
        <div className="text-3xl font-bold text-gray-900">${currentData.Close.toFixed(2)}</div>
        <div className={`text-lg font-bold ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({percentChange.toFixed(2)}%)
        </div>
      </div>

      {/* Day Range */}
      <div className="flex flex-col justify-center">
        <div className="text-base font-semibold text-gray-800 mb-2">Day Range</div>
        <div className="flex items-center space-x-2">
          <span className="text-base font-bold text-gray-900">${dayLow.toFixed(2)}</span>
          <div className="flex-grow h-2 bg-gray-200 rounded-full">
            <div 
              className="h-2 bg-blue-600 rounded-full"
              style={{ 
                width: `${((currentData.Close - dayLow) / (dayHigh - dayLow)) * 100}%` 
              }}
            />
          </div>
          <span className="text-base font-bold text-gray-900">${dayHigh.toFixed(2)}</span>
        </div>
      </div>

      {/* Trading Info */}
      <div className="flex flex-col space-y-3">
        <div>
          <div className="text-base font-semibold text-gray-800 mb-1">Open</div>
          <div className="text-lg font-bold text-gray-900">${currentData.Open.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-base font-semibold text-gray-800 mb-1">Previous Close</div>
          <div className="text-lg font-bold text-gray-900">${previousData.Close.toFixed(2)}</div>
        </div>
      </div>

      {/* Volume */}
      <div className="flex flex-col space-y-3">
        <div>
          <div className="text-base font-semibold text-gray-800 mb-1">Volume</div>
          <div className="text-lg font-bold text-gray-900">{volume.toLocaleString()}</div>
        </div>
        <div>
          <div className="text-base font-semibold text-gray-800 mb-1">Last Update</div>
          <div className="text-lg font-bold text-gray-900">
            {new Date(currentData.Date).toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockInfoPanel;