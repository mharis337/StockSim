import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle2, Info } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const FeatureSelector = ({ 
  isOpen, 
  onClose, 
  onFeaturesSelected, 
  stockData, 
  modelInfo,  // New prop for model information
  previousFeatures = [] // New prop for previously used features
}) => {
  const [selectedFeatures, setSelectedFeatures] = useState(new Set(previousFeatures));
  const [modelRequirements, setModelRequirements] = useState({
    requiredFeatures: 32, // Default value
    recommendedFeatures: [] // Features that work well with this model
  });
  const [validationStatus, setValidationStatus] = useState({
    isValid: true,
    message: '',
    details: []
  });

  // Feature groups with tooltips explaining their use
  const FEATURE_GROUPS = {
    'Price': {
      features: ['Close', 'High', 'Low', 'Open'],
      description: 'Basic price data points'
    },
    'Volume': {
      features: ['Volume'],
      description: 'Trading volume indicators'
    },
    'Moving Averages': {
      features: ['SMA_20', 'EMA_20', 'MA5'],
      description: 'Trend following indicators'
    },
    'MACD': {
      features: ['MACD', 'MACD_Signal', 'MACD_Hist'],
      description: 'Momentum and trend indicators'
    },
    'Bollinger Bands': {
      features: ['BB_Upper', 'BB_Middle', 'BB_Lower'],
      description: 'Volatility and trend indicators'
    },
    'Momentum': {
      features: ['RSI', 'STOCH_K', 'STOCH_D', 'WILLR', 'ROC'],
      description: 'Momentum and overbought/oversold indicators'
    },
    'Volume Indicators': {
      features: ['OBV', 'AD', 'MFI'],
      description: 'Volume-based trend confirmation indicators'
    },
    'Volatility': {
      features: ['ATR', 'STDDEV'],
      description: 'Market volatility measurements'
    },
    'Trend': {
      features: ['ADX', 'PLUS_DI', 'MINUS_DI'],
      description: 'Trend strength and direction indicators'
    },
    'Support/Resistance': {
      features: ['PIVOT', 'R1', 'S1'],
      description: 'Price levels where the market might reverse'
    }
  };

  // Validate selected features against model requirements
  const validateFeatures = () => {
    const count = selectedFeatures.size;
    const required = modelRequirements.requiredFeatures;
    const status = {
      isValid: true,
      message: '',
      details: []
    };

    if (count === 0) {
      status.isValid = false;
      status.message = 'Please select at least one feature';
    } else if (count < required) {
      status.isValid = false;
      status.message = `Model requires ${required} features (${required - count} more needed)`;
      status.details.push(`The remaining features will be auto-selected based on relevance`);
    } else if (count > required) {
      status.isValid = false;
      status.message = `Model only accepts ${required} features (${count - required} will be trimmed)`;
      status.details.push('Only the first features will be used, in order of selection');
    } else {
      status.message = 'Feature selection valid';
    }

    setValidationStatus(status);
  };

  useEffect(() => {
    validateFeatures();
  }, [selectedFeatures]);

  const handleFeatureToggle = (feature) => {
    const newSelected = new Set(selectedFeatures);
    if (newSelected.has(feature)) {
      newSelected.delete(feature);
    } else {
      newSelected.add(feature);
    }
    setSelectedFeatures(newSelected);
  };

  const handleConfirm = () => {
    onFeaturesSelected(Array.from(selectedFeatures));
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white px-6 py-4 border-b border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900">Select Features for Analysis</h2>
          <p className="text-gray-600 mt-1">
            Model requires exactly {modelRequirements.requiredFeatures} features
          </p>
        </div>

        <div className="p-6">
          {/* Feature Selection Status */}
          {validationStatus.message && (
            <Alert 
              variant={validationStatus.isValid ? "default" : "destructive"}
              className="mb-4"
            >
              {validationStatus.isValid ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <AlertTriangle className="h-4 w-4" />
              )}
              <AlertDescription>
                <div className="font-medium">{validationStatus.message}</div>
                {validationStatus.details.map((detail, index) => (
                  <div key={index} className="text-sm mt-1 opacity-80">{detail}</div>
                ))}
              </AlertDescription>
            </Alert>
          )}

          {/* Feature Groups */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(FEATURE_GROUPS).map(([group, { features, description }]) => (
              <div key={group} className="border rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <h3 className="font-semibold text-gray-800">{group}</h3>
                  <button 
                    className="text-gray-400 hover:text-gray-600"
                    onClick={() => {}}
                  >
                    <Info className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-sm text-gray-600 mb-3">{description}</p>
                <div className="space-y-2">
                  {features.map(feature => (
                    <div key={feature} className="flex items-center">
                      <label className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={selectedFeatures.has(feature)}
                          onChange={() => handleFeatureToggle(feature)}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-700">
                          {feature}
                          {previousFeatures.includes(feature) && (
                            <span className="ml-2 text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                              Previously Used
                            </span>
                          )}
                        </span>
                      </label>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="mt-6 flex justify-end space-x-4">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
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