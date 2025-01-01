import React, { useState, useEffect } from 'react';
import { AlertCircle } from 'lucide-react';

const REQUIRED_FEATURES = [
  'BB_Upper', 'STDDEV', 'SAR', 'BB_Middle', 'R1', 'BB_Lower', 
  'OBV', 'S1', 'SMA_20', 'AD', 'MACD_Signal', 'EMA_20', 
  'ATR', 'MACD', 'PIVOT'
];

const ALL_FEATURES = [
  'Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'EMA_20',
  'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle',
  'BB_Lower', 'SAR', 'RSI', 'STOCH_K', 'STOCH_D', 'WILLR',
  'ROC', 'OBV', 'AD', 'MFI', 'ATR', 'STDDEV', 'ADX', 
  'PLUS_DI', 'MINUS_DI', 'ICHIMOKU_CONV', 'PIVOT', 'R1', 'S1'
];

// Group features by category
const FEATURE_GROUPS = {
  'Price': ['Close', 'High', 'Low', 'Open'],
  'Volume': ['Volume'],
  'Moving Averages': ['SMA_20', 'EMA_20'],
  'MACD': ['MACD', 'MACD_Signal', 'MACD_Hist'],
  'Bollinger Bands': ['BB_Upper', 'BB_Middle', 'BB_Lower'],
  'Momentum': ['RSI', 'STOCH_K', 'STOCH_D', 'WILLR', 'ROC'],
  'Volume Indicators': ['OBV', 'AD', 'MFI'],
  'Volatility': ['ATR', 'STDDEV', 'SAR'],
  'Trend': ['ADX', 'PLUS_DI', 'MINUS_DI'],
  'Support/Resistance': ['PIVOT', 'R1', 'S1'],
  'Other': ['ICHIMOKU_CONV']
};

interface FeatureSelectorProps {
  onFeaturesSelected: (features: string[]) => void;
  isOpen: boolean;
  onClose: () => void;
  stockData: any[];
}

const FeatureSelector: React.FC<FeatureSelectorProps> = ({
  onFeaturesSelected,
  isOpen,
  onClose,
  stockData
}) => {
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set([...REQUIRED_FEATURES]));
  const [availableFeatures, setAvailableFeatures] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (stockData && stockData.length > 0) {
      const features = new Set(Object.keys(stockData[0]));
      setAvailableFeatures(features);
    }
  }, [stockData]);

  const handleFeatureToggle = (feature: string) => {
    if (REQUIRED_FEATURES.includes(feature)) {
      return; // Don't allow toggling required features
    }
    const newSelected = new Set(selectedFeatures);
    if (newSelected.has(feature)) {
      newSelected.delete(feature);
    } else {
      newSelected.add(feature);
    }
    setSelectedFeatures(newSelected);
  };

  const handleConfirm = () => {
    // Make sure all required features are included
    const finalFeatures = new Set([...REQUIRED_FEATURES, ...selectedFeatures]);
    onFeaturesSelected(Array.from(finalFeatures));
    onClose();
  };

  if (!isOpen) return null;

  const hasAllRequired = REQUIRED_FEATURES.every(f => availableFeatures.has(f));
  const missingFeatures = REQUIRED_FEATURES.filter(f => !availableFeatures.has(f));

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Select Features for Analysis</h2>

          {missingFeatures.length > 0 && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
              <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 mr-2 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-red-800">Missing Required Features</h3>
                <p className="text-red-700 text-sm mt-1">
                  The following required features are missing: {missingFeatures.join(', ')}
                </p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(FEATURE_GROUPS).map(([group, features]) => (
              <div key={group} className="border rounded-lg p-4">
                <h3 className="font-semibold text-gray-800 mb-3">{group}</h3>
                <div className="space-y-2">
                  {features.map(feature => {
                    const isAvailable = availableFeatures.has(feature);
                    const isRequired = REQUIRED_FEATURES.includes(feature);
                    const isSelected = selectedFeatures.has(feature);

                    return (
                      <div 
                        key={feature}
                        className={`flex items-center ${!isAvailable && 'opacity-50'}`}
                      >
                        <label className="flex items-center space-x-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={isSelected || isRequired}
                            onChange={() => handleFeatureToggle(feature)}
                            disabled={!isAvailable || isRequired}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <span className="text-sm text-gray-700">
                            {feature}
                            {isRequired && (
                              <span className="ml-2 text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                                Required
                              </span>
                            )}
                          </span>
                        </label>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 flex justify-end space-x-4">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              disabled={!hasAllRequired}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              Confirm Selection
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureSelector;