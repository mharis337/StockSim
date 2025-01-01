import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

const SignalPanel = ({ signal }) => {
  if (!signal) {
    return null;
  }

  const getPriceChangePercent = () => {
    if (!signal.price || !signal.target_price) return 0;
    return ((signal.target_price - signal.price) / signal.price) * 100;
  };

  const getSignalColor = (type) => {
    if (!type) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    
    switch (type.toLowerCase()) {
      case 'buy':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'sell':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">ML Signal Analysis</h3>
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${getSignalColor(signal.signal)} border`}>
          {(signal.signal || 'HOLD').toUpperCase()}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-gray-600">Current Price</div>
          <div className="text-xl font-bold">${(signal.price || 0).toFixed(2)}</div>
        </div>
        <div>
          <div className="text-sm text-gray-600">Target Price</div>
          <div className="flex items-center gap-2">
            <div className="text-xl font-bold">${(signal.target_price || 0).toFixed(2)}</div>
            <div className={`text-sm font-medium ${getPriceChangePercent() >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {getPriceChangePercent() >= 0 ? (
                <TrendingUp className="w-4 h-4 inline" />
              ) : (
                <TrendingDown className="w-4 h-4 inline" />
              )}
              {Math.abs(getPriceChangePercent()).toFixed(2)}%
            </div>
          </div>
        </div>
      </div>

      {signal.stop_loss && (
        <div>
          <div className="text-sm text-gray-600">Stop Loss</div>
          <div className="text-lg font-semibold text-red-600">${signal.stop_loss.toFixed(2)}</div>
        </div>
      )}

      <div>
        <div className="text-sm text-gray-600 mb-1">Signal Confidence</div>
        <div className="relative h-2 bg-gray-200 rounded">
          <div 
            className={`absolute h-2 rounded ${signal.signal === 'buy' ? 'bg-green-500' : 'bg-red-500'}`}
            style={{ width: `${signal.confidence || 0}%` }}
          />
        </div>
        <div className="text-sm font-medium mt-1">{signal.confidence || 0}%</div>
      </div>

      <div className="border-t pt-4">
        <div className="text-sm font-medium mb-2">Trade Parameters</div>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Potential Profit:</span>
            <span className="ml-2 font-medium text-green-600">
              ${((signal.target_price || 0) - (signal.price || 0)).toFixed(2)}
            </span>
          </div>
          {signal.stop_loss && (
            <div>
              <span className="text-gray-600">Max Loss:</span>
              <span className="ml-2 font-medium text-red-600">
                ${((signal.price || 0) - (signal.stop_loss || 0)).toFixed(2)}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SignalPanel;
