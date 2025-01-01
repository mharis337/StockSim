import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Play, Trash2, AlertCircle, TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Scatter
} from 'recharts';
import FeatureSelector from './FeatureSelection';
import * as d3 from 'd3';
import SignalPanel from './SignalPanel';

interface Model {
    _id?: string;
    id: string;
    name: string;
    uploadDate: string;
    features?: string[];
    status: 'active' | 'inactive';
    targetSymbol?: string;
    path?: string;
}

interface StockData {
    Date: string;
    timestamp: number;
    Close: number;
    High: number;
    Low: number;
    Open: number;
    Volume: number;
    [key: string]: any;
    buySignal?: boolean;
    sellSignal?: boolean;
}

interface Prediction {
    date: string;
    price: number;
    predicted_price: number;
    predicted_change: number;
    signal: 'buy' | 'sell' | 'hold';
}


interface InfoMessage {
    type: 'success' | 'warning' | 'error';
    message: string;
    details?: string[];
    featureChanges?: {
        added: string[];
        removed: string[];
    };
}



const COLORS = [
    '#4F46E5',
    '#EF4444',
    '#22C55E',
    '#F59E0B',
    '#3B82F6',
    '#8B5CF6',
    '#EC4899',
    '#F97316',
    '#14B8A6',
    '#6B7280',
];

const SignalIndicator = ({ signal }: { signal: string }) => {
    const colors = {
        buy: 'bg-green-100 text-green-800 border-green-200',
        sell: 'bg-red-100 text-red-800 border-red-200',
        hold: 'bg-yellow-100 text-yellow-800 border-yellow-200'
    };

    return (
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${colors[signal]} border`}>
            {signal.toUpperCase()}
        </div>
    );
};

const PredictionAnalysis = ({ predictions }: { predictions: Prediction[] }) => {
    if (!predictions || predictions.length === 0) return null;

    const latestPrediction = predictions[predictions.length - 1];

    return (
        <div className="bg-white p-4 rounded-lg shadow space-y-4">
            <h3 className="text-lg font-semibold">Latest Prediction</h3>
            
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <div className="text-sm text-gray-600">Current Price</div>
                    <div className="text-lg font-bold">${latestPrediction.price.toFixed(2)}</div>
                </div>
                <div>
                    <div className="text-sm text-gray-600">Predicted Price</div>
                    <div className="text-lg font-bold">${latestPrediction.predicted_price.toFixed(2)}</div>
                </div>
            </div>
            
            <div>
                <div className="text-sm text-gray-600 mb-2">Predicted Change</div>
                <div className={`text-lg font-bold ${
                    latestPrediction.predicted_change > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                    {latestPrediction.predicted_change > 0 ? '+' : ''}
                    {latestPrediction.predicted_change.toFixed(2)}%
                </div>
            </div>
            
            <div>
                <div className="text-sm text-gray-600 mb-2">Recommended Action</div>
                <SignalIndicator signal={latestPrediction.signal} />
            </div>
        </div>
    );
};

const ModelManager = () => {
    const [models, setModels] = useState<Model[]>([]);
    const [selectedModel, setSelectedModel] = useState<Model | null>(null);
    const [stockData, setStockData] = useState<StockData[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['Close']);
    const [searchSymbol, setSearchSymbol] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [xDomain, setXDomain] = useState<[number, number] | null>(null);
    const [dropdownOpen, setDropdownOpen] = useState<{ [key: string]: boolean }>({});
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [info, setInfo] = useState<InfoMessage | null>(null); // Add this new state

    const chartContainerRef = useRef<HTMLDivElement | null>(null);
    const [isFeatureSelectorOpen, setFeatureSelectorOpen] = useState(false);
    const [pendingAnalysis, setPendingAnalysis] = useState<{
        model: Model;
        symbol: string;
    } | null>(null);

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

    const renderUploadSection = () => (
        <div className="bg-white rounded-lg shadow p-6 mt-6">
            <h3 className="text-lg font-semibold mb-4">Upload Model</h3>
            <div className="space-y-4">
                <div className="flex items-center justify-center w-full">
                    <label htmlFor="model-upload" className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                            <svg className="w-8 h-8 mb-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                                <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                            </svg>
                            <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                            <p className="text-xs text-gray-500">H5, KERAS, PKL, or JOBLIB files</p>
                        </div>
                        <input
                            id="model-upload"
                            type="file"
                            className="hidden"
                            accept=".h5,.keras,.pkl,.joblib"
                            onChange={async (e) => {
                                if (e.target.files && e.target.files[0]) {
                                    setIsLoading(true);
                                    setError(null);

                                    try {
                                        const formData = new FormData();
                                        formData.append('model', e.target.files[0]);

                                        const token = localStorage.getItem('token');
                                        const response = await fetch('http://localhost:5000/api/models/upload', {
                                            method: 'POST',
                                            headers: {
                                                'Authorization': `Bearer ${token}`
                                            },
                                            body: formData
                                        });

                                        if (!response.ok) {
                                            throw new Error('Failed to upload model');
                                        }

                                        fetchModels();

                                    } catch (err) {
                                        setError('Failed to upload model: ' + (err instanceof Error ? err.message : 'Unknown error'));
                                    } finally {
                                        setIsLoading(false);
                                        e.target.value = '';
                                    }
                                }
                            }}
                        />
                    </label>
                </div>

                {isLoading && (
                    <div className="text-center text-gray-500">
                        <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2" />
                        Uploading model...
                    </div>
                )}

                {error && (
                    <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}
            </div>
        </div>
    );

    const DEFAULT_MODEL_INFO = {
        requiredFeatures: 32, // Default number of features if not specified
        recommendedFeatures: [],
        modelType: 'default',
        features: []
      };

    const handleAnalyzeClick = async (model, symbol) => {
        try {
          setIsLoading(true);
          setError(null);
          
          // Debug log the initial model data
          console.log('Initial model data:', model);
          
          const token = localStorage.getItem('token');
          const modelInfoResponse = await fetch(`http://localhost:5000/api/models/${model.id}/info`, {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          });
          
          if (!modelInfoResponse.ok) {
            console.error('Model info response not OK:', modelInfoResponse.status);
            throw new Error('Failed to fetch model information');
          }
          
          const rawModelInfo = await modelInfoResponse.json();
          console.log('Raw API response:', rawModelInfo);
      
          // If we got an empty response or no required_features, use defaults
          if (!rawModelInfo || typeof rawModelInfo.required_features !== 'number') {
            console.warn('Using default model info due to invalid API response');
            
            setPendingAnalysis({
              model,
              symbol,
              modelInfo: DEFAULT_MODEL_INFO
            });
          } else {
            const processedModelInfo = {
              requiredFeatures: rawModelInfo.required_features,
              recommendedFeatures: rawModelInfo.recommended_features || [],
              modelType: rawModelInfo.model_type || 'unknown',
              features: rawModelInfo.features || []
            };
      
            console.log('Processed model info:', processedModelInfo);
      
            setPendingAnalysis({
              model,
              symbol,
              modelInfo: processedModelInfo
            });
          }
          
          // Debug log the final state
          console.log('Setting pendingAnalysis:', pendingAnalysis);
          
          setFeatureSelectorOpen(true);
          
        } catch (err) {
          console.error('Analysis error:', err);
          setError(`Failed to start analysis: ${err.message}`);
          
          // Even on error, we'll use default values to allow the feature selector to open
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
      

    const REQUIRED_FEATURES = [
        'BB_Upper', 'STDDEV', 'SAR', 'BB_Middle', 'R1', 'BB_Lower', 
        'OBV', 'S1', 'SMA_20', 'AD', 'MACD_Signal', 'EMA_20', 
        'ATR', 'MACD', 'PIVOT'
    ];

    const handleFeaturesSelected = async (selectedFeatures) => {
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
          setError(err.message);
        } finally {
          setIsLoading(false);
          setPendingAnalysis(null);
        }
      };

    const renderInfoMessage = () => {
        if (!info) return null;

        return (
            <Alert 
                variant={info.type === 'success' ? 'default' : info.type === 'warning' ? 'default' : 'destructive'}
                className="mb-4"
            >
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                    <div className="font-medium">{info.message}</div>
                    {info.details && (
                        <ul className="mt-2 ml-4 list-disc">
                            {info.details.map((detail, index) => (
                                <li key={index} className="text-sm">{detail}</li>
                            ))}
                        </ul>
                    )}
                    {info.featureChanges && (
                        <div className="mt-2">
                            {info.featureChanges.added.length > 0 && (
                                <div className="text-sm text-green-600">
                                    Added: {info.featureChanges.added.join(', ')}
                                </div>
                            )}
                            {info.featureChanges.removed.length > 0 && (
                                <div className="text-sm text-red-600">
                                    Removed: {info.featureChanges.removed.join(', ')}
                                </div>
                            )}
                        </div>
                    )}
                </AlertDescription>
            </Alert>
        );
    };




    const indicatorGroups: { [key: string]: string[] } = {
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

            const processedData: StockData[] = data.data.map((item: any) => ({
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

    useEffect(() => {
        if (!chartContainerRef.current || !xDomain) return;

        const svg = d3.select(chartContainerRef.current).select('svg');
        const width = 800;
        const height = 400;
        const margin = { top: 20, right: 60, bottom: 70, left: 60 };

        const xScale = d3.scaleTime()
            .domain([new Date(xDomain[0]), new Date(xDomain[1])])
            .range([margin.left, width - margin.right]);

        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([1, 20])
            .translateExtent([[margin.left, 0], [width - margin.right, height]])
            .extent([[margin.left, 0], [width - margin.right, height]])
            .on("zoom", (event) => {
                const newXScale = event.transform.rescaleX(xScale);
                const newDomain: [number, number] = [
                    newXScale.invert(margin.left).getTime(),
                    newXScale.invert(width - margin.right).getTime(),
                ];
                const dataMin = Math.min(...stockData.map(d => d.timestamp));
                const dataMax = Math.max(...stockData.map(d => d.timestamp));
                const clampedDomain: [number, number] = [
                    Math.max(newDomain[0], dataMin),
                    Math.min(newDomain[1], dataMax),
                ];
                setXDomain(clampedDomain);
            });

        svg.call(zoom as any);

        return () => {
            svg.on(".zoom", null);
        };
    }, [xDomain, stockData]);

    const renderSidePanel = () => {
        if (!stockData.length) return null;

        const latestData = stockData[stockData.length - 1];
        const previousData = stockData[stockData.length - 2];

        const priceChange = latestData.Close - previousData.Close;
        const percentChange = (priceChange / previousData.Close) * 100;

        return (
            <div className="w-64 bg-white p-4 rounded-lg shadow">
                <h3 className="text-xl font-bold mb-4">{searchSymbol.toUpperCase()}</h3>

                <div className="space-y-4">
                    <div>
                        <div className="text-sm text-gray-600">Price</div>
                        <div className="text-2xl font-bold">${latestData.Close.toFixed(2)}</div>
                        <div className={`flex items-center ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {priceChange >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                            <span>{priceChange.toFixed(2)} ({percentChange.toFixed(2)}%)</span>
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

    const renderIndicatorSelector = () => (
        <div className="mb-4 flex flex-wrap gap-4">
            {Object.entries(indicatorGroups).map(([group, indicators]) => (
                <div key={group} className="relative">
                    <button
                        onClick={() => setDropdownOpen(prev => ({ ...prev, [group]: !prev[group] }))}
                        className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center inline-flex items-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
                        type="button"
                    >
                        {group}
                        <svg className="w-2.5 h-2.5 ml-3" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 10 6">
                            <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m1 1 4 4 4-4" />
                        </svg>
                    </button>

                    {dropdownOpen[group] && (
                        <div id={`dropdown-${group}`} className="absolute z-10 bg-white divide-y divide-gray-100 rounded-lg shadow w-56 mt-2">
                            <ul className="py-2 text-sm text-gray-700" aria-labelledby="dropdownDefaultButton">
                                {indicators.map((indicator) => (
                                    <li key={indicator}>
                                        <label className="flex items-center px-4 py-2 hover:bg-gray-100 cursor-pointer">
                                            <input
                                                type="checkbox"
                                                className="form-checkbox h-4 w-4 text-blue-600"
                                                checked={selectedIndicators.includes(indicator)}
                                                onChange={(e) => {
                                                    if (e.target.checked) {
                                                        setSelectedIndicators([...selectedIndicators, indicator]);
                                                    } else {
                                                        setSelectedIndicators(selectedIndicators.filter(i => i !== indicator));
                                                    }
                                                }}
                                            />
                                            <span className="ml-2">{indicator.replace(/_/g, ' ')}</span>
                                        </label>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );

    const getColor = (index: number) => COLORS[index % COLORS.length];

    const renderChart = () => {
        if (selectedIndicators.length === 0) {
            return (
                <div className="h-96 flex items-center justify-center text-gray-500">
                    Please select at least one indicator to display.
                </div>
            );
        }

        const filteredData = xDomain
            ? stockData.filter(d => d.timestamp >= xDomain[0] && d.timestamp <= xDomain[1])
            : stockData;

        const yValues = selectedIndicators.flatMap(indicator =>
            filteredData.map(d => d[indicator as keyof StockData] as number).filter(v => v !== undefined)
        );

        const yDomain = yValues.length > 0 ? [Math.min(...yValues) * 0.95, Math.max(...yValues) * 1.05] : ['auto', 'auto'];

        return (
            <div ref={chartContainerRef}>
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart
                        data={filteredData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                            type="number"
                            dataKey="timestamp"
                            domain={xDomain ? xDomain : ['auto', 'auto']}
                            tickFormatter={(timestamp) => {
                                const date = new Date(timestamp);
                                return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
                            }}
                            scale="linear"
                        />
                        <YAxis
                            yAxisId="left"
                            domain={yDomain}
                            tickFormatter={(value: number) => value?.toFixed(2)}
                            label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip
                            labelFormatter={(label: number) => new Date(label).toLocaleString()}
                            formatter={(value: any, name: string) => {
                                if (value === null || value === undefined) return ['N/A', name];
                                return [value.toFixed(2), name.replace(/_/g, ' ')];
                            }}
                        />
                        <Legend />

                        {selectedIndicators.map((indicator, index) => (
                            <Line
                                key={indicator}
                                type="monotone"
                                dataKey={indicator as keyof StockData}
                                stroke={getColor(index)}
                                dot={false}
                                yAxisId="left"
                                connectNulls
                                name={indicator.replace(/_/g, ' ')}
                            />
                        ))}

                        {selectedIndicators.includes('MACD') && (
                            <>
                                <Line
                                    type="monotone"
                                    dataKey="MACD_Signal"
                                    stroke={getColor(selectedIndicators.indexOf('MACD') + 1)}
                                    dot={false}
                                    yAxisId="left"
                                    name="MACD Signal"
                                />
                                <Line
                                    type="monotone"
                                    dataKey="MACD_Hist"
                                    stroke={getColor(selectedIndicators.indexOf('MACD') + 2)}
                                    dot={false}
                                    yAxisId="left"
                                    name="MACD Histogram"
                                />
                            </>
                        )}

                        <Scatter
                            name="Buy Signal"
                            data={filteredData.filter(d => d.buySignal)}
                            fill="#22C55E"
                            shape="triangle"
                            yAxisId="left"
                            line={{ stroke: '#22C55E' }}
                        />
                        <Scatter
                            name="Sell Signal"
                            data={filteredData.filter(d => d.sellSignal)}
                            fill="#EF4444"
                            shape="triangle"
                            yAxisId="left"
                            line={{ stroke: '#EF4444' }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        );
    };

    return (
        <div className="p-6 bg-gray-50 min-h-screen">
            <div className="max-w-7xl mx-auto space-y-6">
                <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                    <h2 className="text-2xl font-bold text-gray-800">AI Model Manager</h2>
                    <div className="flex items-center space-x-2 w-full md:w-auto">
                        <input
                            type="text"
                            value={searchSymbol}
                            onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                            className="flex-1 p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                            placeholder="Enter stock symbol (e.g., AAPL)"
                            aria-label="Stock Symbol"
                        />
                        <button
                            onClick={handleSearch}
                            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                        >
                            Search
                        </button>
                    </div>
                </div>

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

                {error && (
                    <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}

                {renderInfoMessage()}

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex gap-6">
                        <div className="flex-1">
                            {renderIndicatorSelector()}
                            {isLoading ? (
                                <div className="flex justify-center items-center h-96">
                                    <div className="w-6 h-6 border-4 border-t-4 border-gray-200 border-t-blue-600 rounded-full animate-spin"></div>
                                </div>
                            ) : stockData.length > 0 ? (
                                renderChart()
                            ) : (
                                <div className="h-96 flex items-center justify-center text-gray-500">
                                    No data available. Please search for a stock symbol.
                                </div>
                            )}
                        </div>
                        {renderSidePanel()}
                    </div>
                </div>

                {predictions.length > 0 && (
                    <div className="mt-6">
                        <SignalPanel signal={predictions[predictions.length - 1]} />
                    </div>
                )}

                <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Available Models</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {models.map((model) => (
                            <div
                                key={model.id}
                                className={`p-4 rounded-lg border ${
                                    selectedModel?.id === model.id
                                        ? 'border-blue-500 bg-blue-50'
                                        : 'border-gray-200'
                                }`}
                            >
                                <div className="flex justify-between items-start">
                                    <div>
                                        <h4 className="font-medium">{model.name}</h4>
                                        <p className="text-sm text-gray-600">
                                            Uploaded: {new Date(model.uploadDate).toLocaleDateString()}
                                        </p>
                                    </div>
                                    <div className="flex gap-2">
                                        <button
                                            onClick={() => {
                                                const modelId = model.id || model._id?.toString();
                                                setSelectedModel(model);
                                                if (searchSymbol) {
                                                    handleAnalyzeClick({
                                                        ...model,
                                                        id: modelId
                                                    }, searchSymbol);
                                                }
                                            }}
                                            className={`p-2 rounded ${
                                                selectedModel?.id === model.id
                                                    ? 'bg-blue-600 text-white'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                            }`}
                                            aria-label={`Select model ${model.name}`}
                                        >
                                            <Play className="w-4 h-4" />
                                        </button>
                                        <button
                                            onClick={async () => {
                                                if (window.confirm('Are you sure you want to delete this model?')) {
                                                    try {
                                                        const token = localStorage.getItem('token');
                                                        const response = await fetch(`http://localhost:5000/api/models/${model.id}`, {
                                                            method: 'DELETE',
                                                            headers: {
                                                                'Authorization': `Bearer ${token}`
                                                            }
                                                        });

                                                        if (!response.ok) {
                                                            throw new Error('Failed to delete model');
                                                        }

                                                        setModels(models.filter(m => m.id !== model.id));
                                                        if (selectedModel?.id === model.id) {
                                                            setSelectedModel(null);
                                                        }
                                                    } catch (err) {
                                                        setError('Failed to delete model: ' + (err instanceof Error ? err.message : 'Unknown error'));
                                                    }
                                                }
                                            }}
                                            className="p-2 bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
                                            aria-label={`Delete model ${model.name}`}
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {renderUploadSection()}
            </div>
        </div>
    );
};

export default ModelManager;
