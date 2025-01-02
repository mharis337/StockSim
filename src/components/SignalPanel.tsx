import React from 'react';
import { TrendingUp, TrendingDown, Activity, AlertCircle } from 'lucide-react';

const SignalPanel = ({ signal }) => {
  if (!signal) return null;

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

  const getRSIStatus = (rsi) => {
    if (!rsi && rsi !== 0) return { text: 'Unavailable', color: 'text-gray-400' };
    if (rsi > 70) return { text: 'Overbought', color: 'text-red-600' };
    if (rsi < 30) return { text: 'Oversold', color: 'text-green-600' };
    return { text: 'Neutral', color: 'text-gray-600' };
  };

  const getMACDStatus = (macd, signal) => {
    if ((!macd && macd !== 0) || (!signal && signal !== 0)) {
      return { text: 'Unavailable', color: 'text-gray-400' };
    }
    if (macd > signal) return { text: 'Bullish', color: 'text-green-600' };
    if (macd < signal) return { text: 'Bearish', color: 'text-red-600' };
    return { text: 'Neutral', color: 'text-gray-600' };
  };

  const getBBPosition = (price, upper, lower) => {
    if ((!upper && upper !== 0) || (!lower && lower !== 0)) {
      return { text: 'Unavailable', color: 'text-gray-400' };
    }
    if (price > upper) return { text: 'Above Upper Band', color: 'text-red-600' };
    if (price < lower) return { text: 'Below Lower Band', color: 'text-green-600' };
    return { text: 'Within Bands', color: 'text-gray-600' };
  };

  const tech = signal.technical_indicators || {};
  const rsiStatus = getRSIStatus(tech.rsi);
  const macdStatus = getMACDStatus(tech.macd, tech.macd_signal);
  const bbStatus = getBBPosition(signal.price, tech.bb_upper, tech.bb_lower);

  const predictedPrice = signal.price * (1 + (signal.predicted_change || 0) / 100);

  const formatValue = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return 'N/A';
    }
    return typeof value === 'number' ? value.toFixed(2) : value;
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Signal Analysis</h3>
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${getSignalColor(signal.signal)} border`}>
          {(signal.signal || 'HOLD').toUpperCase()}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-gray-600">Current Price</div>
          <div className="text-xl font-bold">${formatValue(signal.price)}</div>
        </div>
        <div>
          <div className="text-sm text-gray-600">Predicted Change</div>
          <div className="flex items-center gap-2">
            <div className={`text-xl font-bold ${signal.predicted_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {signal.predicted_change >= 0 ? '+' : ''}{formatValue(signal.predicted_change)}%
            </div>
            {signal.predicted_change >= 0 ? (
              <TrendingUp className="w-5 h-5 text-green-600" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-600" />
            )}
          </div>
          <div className="text-sm text-gray-500 mt-1">
            (${formatValue(predictedPrice)} predicted)
          </div>
        </div>
      </div>

      <div>
        <div className="text-sm text-gray-600 mb-1">Signal Confidence</div>
        <div className="relative h-2 bg-gray-200 rounded">
          <div 
            className={`absolute h-2 rounded ${signal.signal === 'buy' ? 'bg-green-500' : 'bg-red-500'}`}
            style={{ width: `${signal.confidence || 0}%` }}
          />
        </div>
        <div className="text-sm font-medium mt-1">{formatValue(signal.confidence)}%</div>
      </div>

      <div className="border-t pt-4">
        <div className="text-sm font-medium mb-2">Technical Indicators</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-3 bg-gray-50 rounded">
            <div className="text-sm text-gray-600">RSI</div>
            <div className="font-semibold">{formatValue(tech.rsi)}</div>
            <div className={`text-xs ${rsiStatus.color}`}>{rsiStatus.text}</div>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <div className="text-sm text-gray-600">MACD</div>
            <div className="font-semibold">{formatValue(tech.macd)}</div>
            <div className={`text-xs ${macdStatus.color}`}>{macdStatus.text}</div>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <div className="text-sm text-gray-600">Bollinger Bands</div>
            <div className="font-semibold">${formatValue(signal.price)}</div>
            <div className={`text-xs ${bbStatus.color}`}>{bbStatus.text}</div>
          </div>
        </div>
      </div>

      {signal.signal !== 'hold' && signal.stop_loss && (
        <div className="border-t pt-4">
          <div className="flex items-center gap-2 text-sm font-medium text-red-600">
            <AlertCircle className="w-4 h-4" />
            Stop Loss: ${formatValue(signal.stop_loss)}
          </div>
        </div>
      )}

      {tech.rsi !== null && tech.macd !== null && (
        <div className="border-t pt-4">
          <div className="text-sm font-medium mb-2">Additional Analysis</div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">MACD Signal Line</span>
              <span className="font-medium">{formatValue(tech.macd_signal)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">BB Upper Band</span>
              <span className="font-medium">${formatValue(tech.bb_upper)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">BB Lower Band</span>
              <span className="font-medium">${formatValue(tech.bb_lower)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SignalPanel;