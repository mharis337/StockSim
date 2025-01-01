import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, PlayCircle, PauseCircle, RefreshCw, Save, Settings, TrendingUp } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const ModelTraining = () => {
  const [symbol, setSymbol] = useState('');
  const [modelConfig, setModelConfig] = useState({
    epochs: 50,
    batch_size: 32,
    sequence_length: 60,
    learning_rate: 0.001
  });
  const [trainingStatus, setTrainingStatus] = useState({
    isTraining: false,
    currentEpoch: 0,
    loss: [],
    validation_loss: []
  });
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [savedModels, setSavedModels] = useState([]);
  const [showConfig, setShowConfig] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);

  useEffect(() => {
    fetchSavedModels();
  }, []);

  const fetchSavedModels = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/api/models', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      setSavedModels(data.models);
    } catch (err) {
      setError('Failed to load saved models');
    }
  };

  const handleStartTraining = async () => {
    if (!symbol) {
      setError('Please enter a stock symbol');
      return;
    }

    setError('');
    setTrainingStatus({
      isTraining: true,
      currentEpoch: 0,
      loss: [],
      validation_loss: []
    });

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/api/model/train', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          symbol,
          config: modelConfig
        })
      });

      if (!response.ok) throw new Error('Training failed');
      const data = await response.json();
      
      monitorTraining(data.model_id);
    } catch (err) {
      setError(err.message);
      setTrainingStatus(prev => ({ ...prev, isTraining: false }));
    }
  };

  const monitorTraining = (modelId) => {
    const eventSource = new EventSource(
      `http://localhost:5000/api/model/training-status/${modelId}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.error) {
        setError(data.error);
        eventSource.close();
        return;
      }

      if (data.status === 'completed') {
        eventSource.close();
        setTrainingStatus(prev => ({ ...prev, isTraining: false }));
        fetchSavedModels();
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setError('Lost connection to training server');
      setTrainingStatus(prev => ({ ...prev, isTraining: false }));
    };
  };

  const handleMakePrediction = async (modelId) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/api/model/predict', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_id: modelId,
          symbol: symbol
        })
      });

      if (!response.ok) throw new Error('Prediction failed');
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError('Failed to make prediction');
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="bg-white/50 backdrop-blur-sm rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">Train AI Model</h2>
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Settings className="w-5 h-5" />
            Configure Model
          </button>
        </div>

        {showConfig && (
          <div className="mb-6 p-4 bg-white rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Model Configuration</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { label: "Epochs", key: "epochs", min: 1, max: 1000 },
                { label: "Batch Size", key: "batch_size", min: 1 },
                { label: "Sequence Length", key: "sequence_length", min: 10, max: 200 },
                { label: "Learning Rate", key: "learning_rate", step: "0.0001", min: 0.0001, max: 0.1 }
              ].map((field) => (
                <div key={field.key}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {field.label}
                  </label>
                  <input
                    type="number"
                    value={modelConfig[field.key]}
                    onChange={(e) => setModelConfig(prev => ({
                      ...prev,
                      [field.key]: field.key === "learning_rate" ? 
                        parseFloat(e.target.value) : 
                        parseInt(e.target.value)
                    }))}
                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                  />
                </div>
              ))}
            </div>
            
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <h4 className="text-sm font-semibold text-blue-800 mb-2">Model Architecture</h4>
              <p className="text-sm text-blue-600">
                LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
              </p>
            </div>
          </div>
        )}

        <div className="mb-6">
          <div className="flex gap-4">
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="Enter stock symbol (e.g., AAPL)..."
              className="flex-1 p-2 border rounded focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleStartTraining}
              disabled={!symbol || trainingStatus.isTraining}
              className={`flex items-center gap-2 px-6 py-2 rounded font-medium transition-colors ${
                trainingStatus.isTraining
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {trainingStatus.isTraining ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <PlayCircle className="w-5 h-5" />
                  Start Training
                </>
              )}
            </button>
          </div>
        </div>

        {trainingStatus.isTraining && (
          <div className="bg-white p-4 rounded-lg border border-gray-200 mb-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <RefreshCw className="w-5 h-5 animate-spin text-blue-600" />
                <span className="font-medium text-gray-700">Training in Progress</span>
              </div>
            </div>

            <div className="text-sm text-gray-600">
              Model is being trained on historical data for {symbol}...
            </div>
          </div>
        )}

        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="mt-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Saved Models</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {savedModels.map((model, index) => (
              <div
                key={`model-${model.id || index}`}
                className="p-4 border border-gray-200 rounded-lg bg-white hover:shadow-md transition-shadow"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-medium text-gray-900">{model.symbol}</h4>
                    <p className="text-sm text-gray-500">
                      Created: {new Date(model.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleMakePrediction(model.id)}
                      className="text-blue-600 hover:text-blue-700"
                      title="Make Prediction"
                    >
                      <TrendingUp className="w-5 h-5" />
                    </button>
                  </div>
                </div>
                
                <div className="mt-2">
                  <div className="text-sm text-gray-600">
                    Accuracy: {model.accuracy ? `${(model.accuracy * 100).toFixed(2)}%` : 'N/A'}
                  </div>
                  <div className="text-sm text-gray-600">
                    Status: <span className={`font-medium ${
                      model.status === 'completed' ? 'text-green-600' : 'text-blue-600'
                    }`}>{model.status}</span>
                  </div>
                </div>
              </div>
            ))}

            {savedModels.length === 0 && (
              <div className="col-span-full text-center py-8 text-gray-500">
                No trained models yet. Start by training a new model.
              </div>
            )}
          </div>
        </div>

        {prediction && (
          <div className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200">
            <h4 className="text-lg font-semibold text-green-800 mb-2">Prediction Results</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-green-600">Current Price</p>
                <p className="text-lg font-bold text-green-900">
                  ${prediction.current_price.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-green-600">Predicted Price ({prediction.prediction_date})</p>
                <p className="text-lg font-bold text-green-900">
                  ${prediction.prediction.toFixed(2)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelTraining;
