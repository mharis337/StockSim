import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';
import { SearchBar } from './SeachModelManager';
import { ModelUpload } from './ModelUpload';
import { ModelList } from './ModelList';
import { StockChart } from './ModelChart';
import { IndicatorSelector } from './IndicatorSelector';
import FeatureSelector from './FeatureSelection';
import SignalPanel from './SignalPanel';
import BacktestPanel from './BacktestPanel';
import { StockSidePanel } from './StockSidePanel';

interface Model {
    id: string;
    _id?: string;
    name: string;
    uploadDate: string;
    features?: string[];
    status: 'active' | 'inactive';
}

const DEFAULT_MODEL_INFO = {
    requiredFeatures: 32,
    recommendedFeatures: [],
    modelType: 'default',
    features: []
};

const indicatorGroups = {
    'Price': ['Close', 'High', 'Low', 'Open'],
    'Volume': ['Volume'],
    'Moving Averages': ['SMA_20', 'EMA_20'],
    'MACD': ['MACD', 'MACD_Signal', 'MACD_Hist'],
    'Bollinger Bands': ['BB_Upper', 'BB_Middle', 'BB_Lower'],
    'Momentum': ['RSI', 'STOCH_K', 'STOCH_D', 'WILLR', 'ROC'],
    'Volume Indicators': ['OBV', 'AD', 'MFI'],
    'Volatility': ['ATR', 'STDDEV'],
    'Trend': ['ADX', 'PLUS_DI', 'MINUS_DI'],
    'Others': ['ICHIMOKU_CONV', 'PIVOT', 'R1', 'S1']
};

const ModelManager: React.FC = () => {
    const [models, setModels] = useState<Model[]>([]);
    const [selectedModel, setSelectedModel] = useState<Model | null>(null);
    const [stockData, setStockData] = useState<any[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['Close']);
    const [searchSymbol, setSearchSymbol] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [xDomain, setXDomain] = useState<[number, number] | null>(null);
    const [dropdownOpen, setDropdownOpen] = useState<{ [key: string]: boolean }>({});
    const [predictions, setPredictions] = useState<any[]>([]);
    const [isFeatureSelectorOpen, setFeatureSelectorOpen] = useState(false);
    const [pendingAnalysis, setPendingAnalysis] = useState<{
        model: Model;
        symbol: string;
        modelInfo?: any;
    } | null>(null);

    const chartContainerRef = useRef<HTMLDivElement | null>(null);

    const fetchModels = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:5000/api/models', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) throw new Error('Failed to fetch models');
            const data = await response.json();
            setModels(data.models);
        } catch (err) {
            setError('Failed to load saved models');
        }
    };

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchStockData = useCallback(async (symbol: string) => {
        try {
            setIsLoading(true);
            setError(null);
            setStockData([]);

            const token = localStorage.getItem('token');
            if (!token) throw new Error('Authentication token not found');

            const response = await fetch(
                `http://localhost:5000/api/historic_stock/${symbol}?interval=1d&timeframe=5y`,
                {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                    },
                }
            );

            if (!response.ok) throw new Error(`Failed to fetch stock data: ${response.statusText}`);

            const data = await response.json();
            if (!data.data || !Array.isArray(data.data)) {
                throw new Error('Invalid data format received from API');
            }

            const processedData = data.data.map((item: any) => ({
                ...item,
                timestamp: new Date(item.Date).getTime(),
            }));

            setStockData(processedData);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch data');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        if (stockData.length > 0) {
            const timestamps = stockData.map(d => d.timestamp);
            setXDomain([Math.min(...timestamps), Math.max(...timestamps)]);
        }
    }, [stockData]);

    const handleSearch = () => {
        const trimmedSymbol = searchSymbol.trim().toUpperCase();
        if (trimmedSymbol === '') {
            setError('Please enter a stock symbol.');
            return;
        }
        fetchStockData(trimmedSymbol);
    };

    const handleModelUpload = async (file: File) => {
        setIsLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('model', file);

            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:5000/api/models/upload', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });

            if (!response.ok) throw new Error('Failed to upload model');
            await fetchModels();
        } catch (err) {
            setError('Failed to upload model: ' + (err instanceof Error ? err.message : 'Unknown error'));
        } finally {
            setIsLoading(false);
        }
    };

    const handleDeleteModel = async (model: Model) => {
        if (window.confirm('Are you sure you want to delete this model?')) {
            try {
                const token = localStorage.getItem('token');
                const response = await fetch(`http://localhost:5000/api/models/${model.id}`, {
                    method: 'DELETE',
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (!response.ok) throw new Error('Failed to delete model');

                setModels(models.filter(m => m.id !== model.id));
                if (selectedModel?.id === model.id) {
                    setSelectedModel(null);
                }
            } catch (err) {
                setError('Failed to delete model: ' + (err instanceof Error ? err.message : 'Unknown error'));
            }
        }
    };

    const handleAnalyzeClick = async (model: Model, symbol: string) => {
        try {
            setIsLoading(true);
            setError(null);

            const token = localStorage.getItem('token');
            const modelInfoResponse = await fetch(`http://localhost:5000/api/models/${model.id}/info`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!modelInfoResponse.ok) {
                throw new Error('Failed to fetch model information');
            }

            const rawModelInfo = await modelInfoResponse.json();

            setPendingAnalysis({
                model,
                symbol,
                modelInfo: rawModelInfo || DEFAULT_MODEL_INFO
            });

            setFeatureSelectorOpen(true);
        } catch (err) {
            setError(`Failed to start analysis: ${err instanceof Error ? err.message : 'Unknown error'}`);
            setPendingAnalysis({
                model,
                symbol,
                modelInfo: DEFAULT_MODEL_INFO
            });
            setFeatureSelectorOpen(true);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFeaturesSelected = async (selectedFeatures: string[]) => {
        if (!pendingAnalysis) return;

        const { model, symbol } = pendingAnalysis;
        setFeatureSelectorOpen(false);

        try {
            setIsLoading(true);
            setError(null);

            const token = localStorage.getItem('token');
            const response = await fetch(`http://localhost:5000/api/model/${model.id}/analyze`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol,
                    features: selectedFeatures
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }

            const data = await response.json();
            setStockData(prevData => {
                return prevData.map(record => {
                    const signal = data.signals.find(s =>
                        s.date === new Date(record.Date).toISOString().split('T')[0]
                    );
                    if (signal) {
                        return {
                            ...record,
                            buySignal: signal.signal === 'buy',
                            sellSignal: signal.signal === 'sell',
                            targetPrice: signal.target_price,
                            stopLoss: signal.stop_loss,
                            confidence: signal.confidence
                        };
                    }
                    return record;
                });
            });

            setPredictions(data.signals);

        } catch (err) {
            setError(err instanceof Error ? err.message : 'Analysis failed');
        } finally {
            setIsLoading(false);
            setPendingAnalysis(null);
        }
    };

    return (
        <div className="p-6 bg-gray-50 min-h-screen">
            <div className="max-w-7xl mx-auto space-y-6">
                <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                    <h2 className="text-2xl font-bold text-gray-800">AI Model Manager</h2>
                    <SearchBar
                        searchSymbol={searchSymbol}
                        setSearchSymbol={setSearchSymbol}
                        handleSearch={handleSearch}
                    />
                </div>

                {error && (
                    <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex gap-6">
                        <div className="flex-1">
                            <IndicatorSelector
                                indicatorGroups={indicatorGroups}
                                selectedIndicators={selectedIndicators}
                                setSelectedIndicators={setSelectedIndicators}
                                dropdownOpen={dropdownOpen}
                                setDropdownOpen={setDropdownOpen}
                            />
                            {isLoading ? (
                                <div className="flex justify-center items-center h-96">
                                    <div className="w-6 h-6 border-4 border-t-4 border-gray-200 border-t-blue-600 rounded-full animate-spin"></div>
                                </div>
                            ) : stockData.length > 0 ? (
                                <StockChart
                                    data={stockData}
                                    selectedIndicators={selectedIndicators}
                                    xDomain={xDomain}
                                    chartContainerRef={chartContainerRef}
                                    onZoom={(newDomain) => setXDomain(newDomain)}
                                />
                            ) : (
                                <div className="h-96 flex items-center justify-center text-gray-500">
                                    No data available. Please search for a stock symbol.
                                </div>
                            )}
                        </div>
                        {stockData.length > 0 && (
                            <StockSidePanel
                                symbol={searchSymbol}
                                stockData={stockData}
                            />
                        )}
                    </div>
                </div>

                {predictions.length > 0 && (
                    <div className="mt-6">
                        <SignalPanel signal={predictions[predictions.length - 1]} />
                    </div>
                )}

                {selectedModel && searchSymbol && (
                    <BacktestPanel
                        modelId={selectedModel.id}
                        symbol={searchSymbol}
                        onBacktestComplete={(results) => {
                            console.log('Backtest completed:', results);
                        }}
                    />
                )}

                <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Available Models</h3>
                    <ModelList
                        models={models}
                        selectedModel={selectedModel}
                        onSelectModel={(model) => {
                            const modelId = model.id || model._id?.toString();
                            setSelectedModel(model);
                            if (searchSymbol) {
                                handleAnalyzeClick({
                                    ...model,
                                    id: modelId
                                }, searchSymbol);
                            }
                        }}
                        onDeleteModel={handleDeleteModel}
                    />
                </div>

                <ModelUpload
                    isLoading={isLoading}
                    error={error}
                    onUpload={handleModelUpload}
                />

                {isFeatureSelectorOpen && pendingAnalysis && (
                    <FeatureSelector
                        isOpen={isFeatureSelectorOpen}
                        onClose={() => {
                            setFeatureSelectorOpen(false);
                            setPendingAnalysis(null);
                        }}
                        onFeaturesSelected={handleFeaturesSelected}
                        modelInfo={pendingAnalysis.modelInfo || DEFAULT_MODEL_INFO}
                        previousFeatures={[]}
                    />
                )}
            </div>
        </div>
    );
};

export default ModelManager;