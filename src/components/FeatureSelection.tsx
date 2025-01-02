import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle2, Info, XCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const ValidationModal = ({ message, details, onClose }) => (
  <div className="fixed inset-0 bg-black bg-opacity-50 z-[60] flex items-center justify-center p-4">
    <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
      <div className="flex items-start mb-4">
        <AlertCircle className="h-6 w-6 text-red-500 mr-3 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">{message}</h3>
          {details && (
            <ul className="text-sm text-gray-600 list-disc pl-5 space-y-1">
              {details.map((detail, index) => (
                <li key={index}>{detail}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="mt-6 flex justify-end">
        <button
          onClick={onClose}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Understood
        </button>
      </div>
    </div>
  </div>
);

const FeatureSelector = ({ 
  isOpen, 
  onClose, 
  onFeaturesSelected, 
  modelInfo,
  previousFeatures = [] 
}) => {
  const [selectedFeatures, setSelectedFeatures] = useState(new Set(previousFeatures));
  const [validationError, setValidationError] = useState(null);
  const [showValidationModal, setShowValidationModal] = useState(false);

    useEffect(() => {
      console.log('Full modelInfo object:', modelInfo);
      if (modelInfo) {
        console.log('Required features:', modelInfo.requiredFeatures);
        console.log('Model info keys:', Object.keys(modelInfo));
      }
    }, [modelInfo]);
  
    const getRequiredFeatures = () => {
      if (!modelInfo) return null;
      
      const required = modelInfo.requiredFeatures || 
                      modelInfo.required_features ||
                      (modelInfo.model && modelInfo.model.requiredFeatures) ||
                      (modelInfo.model && modelInfo.model.required_features);
  
      console.log('Extracted required features:', required);
      return required;
    };
  
    const requiredFeatures = getRequiredFeatures();

    if (!modelInfo || typeof requiredFeatures !== 'number') {
      console.error('Invalid model configuration:', {
        modelInfo,
        requiredFeatures
      });
      return null;
    }
  if (typeof requiredFeatures !== 'number' || requiredFeatures <= 0) {
    console.error('Invalid or missing required features count:', modelInfo);
  }

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
      features: ['SMA_20', 'EMA_20'],
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
      features: ['ATR', 'STDDEV', 'SAR'],
      description: 'Market volatility measurements'
    },
    'Trend': {
      features: ['ADX', 'PLUS_DI', 'MINUS_DI', 'ICHIMOKU_CONV'],
      description: 'Trend strength and direction indicators'
    },
    'Support/Resistance': {
      features: ['PIVOT', 'R1', 'S1'],
      description: 'Price levels where the market might reverse'
    }
  };

  const validateFeatures = () => {
    const count = selectedFeatures.size;
    
    if (!modelInfo || typeof requiredFeatures !== 'number') {
      return {
        isValid: false,
        message: 'Model Configuration Error',
        details: [
          'Unable to determine required feature count.',
          'Please try again or contact support if the issue persists.'
        ]
      };
    }

    if (count === 0) {
      return {
        isValid: false,
        message: 'Feature Selection Error',
        details: ['Please select at least one feature']
      };
    }

    if (count < requiredFeatures) {
      return {
        isValid: false,
        message: 'Insufficient Features Selected',
        details: [
          `This model requires exactly ${requiredFeatures} features`,
          `You have selected ${count} features`,
          `Please select ${requiredFeatures - count} more features`
        ]
      };
    }

    if (count > requiredFeatures) {
      return {
        isValid: false,
        message: 'Too Many Features Selected',
        details: [
          `This model requires exactly ${requiredFeatures} features`,
          `You have selected ${count} features`,
          `Please remove ${count - requiredFeatures} features`
        ]
      };
    }

    return { isValid: true };
  };

  const handleFeatureToggle = (feature) => {
    const newSelected = new Set(selectedFeatures);
    if (newSelected.has(feature)) {
      newSelected.delete(feature);
    } else {
      newSelected.add(feature);
    }
    setSelectedFeatures(newSelected);
    setValidationError(null);
  };

  const handleConfirm = () => {
    const validation = validateFeatures();
    if (!validation.isValid) {
      setValidationError(validation);
      setShowValidationModal(true);
      return;
    }
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
            Model requires exactly {requiredFeatures} features
          </p>
        </div>

        <div className="p-6">
          <Alert 
            variant="default"
            className="mb-6"
          >
            <Info className="h-4 w-4" />
            <AlertDescription>
              Selected features: {selectedFeatures.size} / {requiredFeatures} required
            </AlertDescription>
          </Alert>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(FEATURE_GROUPS).map(([group, { features, description }]) => (
              <div key={group} className="border rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <h3 className="font-semibold text-gray-800">{group}</h3>
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

          <div className="mt-6 flex justify-end space-x-4">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Confirm Selection
            </button>
          </div>
        </div>
      </div>

      {showValidationModal && validationError && (
        <ValidationModal
          message={validationError.message}
          details={validationError.details}
          onClose={() => setShowValidationModal(false)}
        />
      )}
    </div>
  );
};

export default FeatureSelector;