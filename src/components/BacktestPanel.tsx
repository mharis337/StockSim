import React, { useState, useEffect } from 'react';
import { RefreshCw, TrendingUp, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface BacktestPanelProps {
    modelId: string;
    symbol: string;
    onBacktestComplete?: (results: any) => void;
}

const BacktestPanel: React.FC<BacktestPanelProps> = ({ modelId, symbol, onBacktestComplete }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [backtestResults, setBacktestResults] = useState<any>(null);
    const [lookbackDays, setLookbackDays] = useState(30);

    useEffect(() => {
        if (!modelId || !symbol) {
            setError(modelId ? 'No symbol provided' : 'No model ID provided');
        } else {
            setError(null);
        }
    }, [modelId, symbol]);

    const runBacktest = async () => {
        try {
            if (!modelId || !symbol) {
                setError('Both model ID and symbol are required');
                return;
            }

            setIsLoading(true);
            setError(null);

            const token = localStorage.getItem('token');
            if (!token) {
                throw new Error('Authentication token not found');
            }

            console.log('Running backtest for:', { modelId, symbol, lookbackDays });

            const response = await fetch(`http://localhost:5000/api/model/${modelId}/backtest`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol,
                    lookback_days: lookbackDays,
                    features: []
                }),
                credentials: 'include'
            });

            let data;
            try {
                data = await response.json();
            } catch (e) {
                throw new Error('Failed to parse server response');
            }

            if (!response.ok) {
                throw new Error(data.detail || data.error || 'Backtest failed');
            }

            setBacktestResults(data);
            onBacktestComplete?.(data);

        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to run backtest');
            console.error('Backtest error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    if (!modelId || !symbol) {
        return (
            <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                    Cannot run backtest: {!modelId ? 'No model selected' : 'No symbol provided'}
                </AlertDescription>
            </Alert>
        );
    }

    return (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
            <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Model Backtest</h3>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <label className="text-sm text-gray-600">Lookback Days:</label>
                        <input
                            type="number"
                            min="7"
                            max="365"
                            value={lookbackDays}
                            onChange={(e) => setLookbackDays(Math.max(7, Math.min(365, Number(e.target.value))))}
                            className="w-20 p-1 border rounded"
                        />
                    </div>
                    <button
                        onClick={runBacktest}
                        disabled={isLoading || !modelId || !symbol}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? (
                            <>
                                <RefreshCw className="w-4 h-4 animate-spin" />
                                Running...
                            </>
                        ) : (
                            <>
                                <TrendingUp className="w-4 h-4" />
                                Run Backtest
                            </>
                        )}
                    </button>
                </div>
            </div>

            {error && (
                <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {backtestResults && (
                <div className="mt-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 bg-gray-50 rounded-lg">
                            <div className="text-sm text-gray-600">Direction Accuracy</div>
                            <div className="text-xl font-bold text-gray-900">
                                {backtestResults.metrics?.direction_accuracy?.toFixed(1)}%
                            </div>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg">
                            <div className="text-sm text-gray-600">Profitability</div>
                            <div className="text-xl font-bold text-gray-900">
                                {backtestResults.metrics?.profitability?.toFixed(1)}%
                            </div>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg">
                            <div className="text-sm text-gray-600">Average Error</div>
                            <div className="text-xl font-bold text-gray-900">
                                {typeof backtestResults.metrics?.average_error === 'number' 
                                    ? `${backtestResults.metrics.average_error.toFixed(2)}%` 
                                    : '0%'}
                            </div>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg">
                            <div className="text-sm text-gray-600">Total Trades</div>
                            <div className="text-xl font-bold text-gray-900">
                                {backtestResults.metrics?.total_trades}
                            </div>
                            <div className="text-sm text-green-600">
                                {backtestResults.metrics?.profitable_trades} profitable
                            </div>
                        </div>
                    </div>

                    {backtestResults.metrics?.lookback_period && (
                        <div className="mt-4 space-y-2">
                            <div className="text-sm text-gray-600">
                                Test Period: {backtestResults.metrics.lookback_period.start} to {backtestResults.metrics.lookback_period.end}
                            </div>
                            {backtestResults.features_used && backtestResults.features_used.length > 0 && (
                                <div className="text-sm text-gray-600">
                                    Features Used: {backtestResults.features_used.join(', ')}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default BacktestPanel;