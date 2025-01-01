import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Play, Trash2, AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';
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
import * as d3 from 'd3';

interface Model {
  _id: string;
  name: string;
  uploadDate: string;
  features?: string[];
  status: 'active' | 'inactive';
  targetSymbol?: string;
}

interface StockData {
  Date: string;
  timestamp: number; // Added timestamp field
  Close: number;
  High: number;
  Low: number;
  Open: number;
  Volume: number;
  SMA_20?: number;
  EMA_20?: number;
  MACD?: number;
  MACD_Signal?: number;
  MACD_Hist?: number;
  BB_Upper?: number;
  BB_Middle?: number;
  BB_Lower?: number;
  SAR?: number;
  RSI?: number;
  STOCH_K?: number;
  STOCH_D?: number;
  WILLR?: number;
  ROC?: number;
  OBV?: number;
  AD?: number;
  MFI?: number;
  ATR?: number;
  STDDEV?: number;
  ADX?: number;
  PLUS_DI?: number;
  MINUS_DI?: number;
  ICHIMOKU_CONV?: number;
  PIVOT?: number;
  R1?: number;
  S1?: number;
  buySignal?: boolean;
  sellSignal?: boolean;
}

const COLORS = [
  '#4F46E5', // Indigo
  '#EF4444', // Red
  '#22C55E', // Green
  '#F59E0B', // Amber
  '#3B82F6', // Blue
  '#8B5CF6', // Purple
  '#EC4899', // Pink
  '#F97316', // Orange
  '#14B8A6', // Teal
  '#6B7280', // Gray
  // Add more colors as needed
];

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

  const chartContainerRef = useRef<HTMLDivElement | null>(null);

  // Technical indicators grouping
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

  // Fetch stock data with technical indicators
  const fetchStockData = useCallback(async (symbol: string) => {
    try {
      setIsLoading(true);
      setError(null);
      setStockData([]); // Optional: Clear previous data

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
      console.log('Raw API response:', data); // Debug log
      
      if (!data.data || !Array.isArray(data.data)) {
        throw new Error('Invalid data format received from API');
      }

      // Convert Date strings to timestamps
      const processedData: StockData[] = data.data.map((item: any) => ({
        ...item,
        timestamp: new Date(item.Date).getTime(), // Add timestamp field
      }));

      setStockData(processedData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
      console.error('Stock data fetch error:', err); // Debug log
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initialize xDomain when stockData changes
  useEffect(() => {
    if (stockData.length > 0) {
      const timestamps = stockData.map(d => d.timestamp);
      setXDomain([Math.min(...timestamps), Math.max(...timestamps)]);
    }
  }, [stockData]);

  // Handle Search Button Click
  const handleSearch = () => {
    const trimmedSymbol = searchSymbol.trim().toUpperCase();
    if (trimmedSymbol === '') {
      setError('Please enter a stock symbol.');
      return;
    }
    fetchStockData(trimmedSymbol);
  };

  // Implement D3 Zoom Behavior
  useEffect(() => {
    if (!chartContainerRef.current || !xDomain) return;

    const svg = d3.select(chartContainerRef.current).select('svg');
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 60, bottom: 70, left: 60 };

    // Define the initial scale
    const xScale = d3.scaleTime()
      .domain([new Date(xDomain[0]), new Date(xDomain[1])])
      .range([margin.left, width - margin.right]);

    // Define the zoom behavior
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
        // Clamp the newDomain to the data's range
        const dataMin = Math.min(...stockData.map(d => d.timestamp));
        const dataMax = Math.max(...stockData.map(d => d.timestamp));
        const clampedDomain: [number, number] = [
          Math.max(newDomain[0], dataMin),
          Math.min(newDomain[1], dataMax),
        ];
        setXDomain(clampedDomain);
      });

    // Apply the zoom behavior to the SVG
    svg.call(zoom as any);

    // Cleanup on unmount
    return () => {
      svg.on(".zoom", null);
    };
  }, [xDomain, stockData]);

  // Side panel with stock details
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
          
          {/* ML Model Signals */}
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

  // Indicator selector with dropdowns
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
              <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m1 1 4 4 4-4"/>
            </svg>
          </button>
          
          {/* Dropdown menu */}
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

  // Assign colors to indicators
  const getColor = (index: number) => COLORS[index % COLORS.length];

  // Render Chart
  const renderChart = () => {
    if (selectedIndicators.length === 0) {
      return (
        <div className="h-96 flex items-center justify-center text-gray-500">
          Please select at least one indicator to display.
        </div>
      );
    }

    // Prepare data within the xDomain
    const filteredData = xDomain
      ? stockData.filter(d => d.timestamp >= xDomain[0] && d.timestamp <= xDomain[1])
      : stockData;

    // Determine Y domains for shared Y-axis
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
              scale="linear" // Changed from "time" to "linear"
            />
            {/* Explicitly set yAxisId="left" */}
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
            
            {/* Render Lines for each selected indicator */}
            {selectedIndicators.map((indicator, index) => (
              <Line
                key={indicator}
                type="monotone"
                dataKey={indicator as keyof StockData}
                stroke={getColor(index)}
                dot={false}
                yAxisId="left" // Ensure yAxisId matches the YAxis
                connectNulls
                name={indicator.replace(/_/g, ' ')}
              />
            ))}

            {/* Special handling for indicators with multiple components */}
            {selectedIndicators.includes('MACD') && (
              <>
                <Line
                  type="monotone"
                  dataKey="MACD_Signal"
                  stroke={getColor(selectedIndicators.indexOf('MACD') + 1)}
                  dot={false}
                  yAxisId="left" // Ensure yAxisId matches the YAxis
                  name="MACD Signal"
                />
                <Line
                  type="monotone"
                  dataKey="MACD_Hist"
                  stroke={getColor(selectedIndicators.indexOf('MACD') + 2)}
                  dot={false}
                  yAxisId="left" // Ensure yAxisId matches the YAxis
                  name="MACD Histogram"
                />
              </>
            )}

            {/* Buy/Sell signals */}
            <Scatter
              name="Buy Signal"
              data={filteredData.filter(d => d.buySignal)}
              fill="#22C55E"
              shape="triangle"
              yAxisId="left" // Ensure yAxisId matches the YAxis
              line={{ stroke: '#22C55E' }}
            />
            <Scatter
              name="Sell Signal"
              data={filteredData.filter(d => d.sellSignal)}
              fill="#EF4444"
              shape="triangle"
              yAxisId="left" // Ensure yAxisId matches the YAxis
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
        {/* Header */}
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

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Main content */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex gap-6">
            {/* Chart section */}
            <div className="flex-1">
              {renderIndicatorSelector()}
              {isLoading ? (
                <div className="flex justify-center items-center h-96">
                  {/* Tailwind CSS Spinner */}
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
            
            {/* Side panel */}
            {renderSidePanel()}
          </div>
        </div>

        {/* Models List */}
        <div className="bg-white rounded-lg shadow p-6 mt-6">
          <h3 className="text-lg font-semibold mb-4">Available Models</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((model) => (
              <div
                key={model._id}
                className={`p-4 rounded-lg border ${
                  selectedModel?._id === model._id
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
                      onClick={() => setSelectedModel(model)}
                      className={`p-2 rounded ${
                        selectedModel?._id === model._id
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                      aria-label={`Select model ${model.name}`}
                    >
                      <Play className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => {
                        // Implement delete functionality
                        // Example:
                        // deleteModel(model._id);
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
      </div>
    </div>
  );
};

export default ModelManager;
