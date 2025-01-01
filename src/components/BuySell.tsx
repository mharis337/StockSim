import React, { useState, useEffect } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface BuySellProps {
    stockData: any[];
    symbol: string;
    initialAction?: 'buy' | 'sell';
    maxSellQuantity?: number;
    onTransactionComplete?: () => void;
}

const roundMoney = (value: number): number => {
    return Number(Math.round(Number(value + 'e2')) + 'e-2');
};

const BuySell: React.FC<BuySellProps> = ({ 
    stockData, 
    symbol, 
    initialAction = 'buy',
    maxSellQuantity = 0,
    onTransactionComplete 
}) => {
    const [quantity, setQuantity] = useState<number>(1);
    const [orderType, setOrderType] = useState<'buy' | 'sell'>(initialAction);
    const [userBalance, setUserBalance] = useState<number>(0);
    const [ownedQuantity, setOwnedQuantity] = useState<number>(0);
    const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isBalanceLoading, setIsBalanceLoading] = useState(true);

    const currentPrice = stockData.length > 0 ? 
        roundMoney(stockData[stockData.length - 1].Close) : 0;

    const totalCost = roundMoney(currentPrice * quantity);

    useEffect(() => {
        setQuantity(1);
        setMessage(null);
    }, [orderType]);

    useEffect(() => {
        const fetchUserData = async () => {
            try {
                const token = localStorage.getItem('token');
                if (!token) throw new Error('No token found');

                const balanceResponse = await fetch('http://localhost:5000/api/user/balance', {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Accept': 'application/json'
                    }
                });

                if (!balanceResponse.ok) throw new Error('Failed to fetch balance');
                const balanceData = await balanceResponse.json();
                setUserBalance(roundMoney(balanceData.balance));

                const portfolioResponse = await fetch('http://localhost:5000/api/portfolio', {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Accept': 'application/json'
                    }
                });

                if (!portfolioResponse.ok) throw new Error('Failed to fetch portfolio');
                const portfolioData = await portfolioResponse.json();
                const holding = portfolioData.portfolio.find(h => h.symbol === symbol);
                setOwnedQuantity(holding ? holding.quantity : 0);

            } catch (error) {
                console.error('Error fetching user data:', error);
                setMessage({
                    type: 'error',
                    text: error instanceof Error ? error.message : 'Failed to load user data'
                });
            } finally {
                setIsBalanceLoading(false);
            }
        };

        fetchUserData();
    }, [symbol]);

    const handleQuantityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newQuantity = Math.max(1, parseInt(e.target.value) || 1);
        if (orderType === 'sell') {
            setQuantity(Math.min(newQuantity, ownedQuantity));
        } else {
            const maxBuyable = Math.floor(userBalance / currentPrice);
            setQuantity(Math.min(newQuantity, maxBuyable));
        }
    };

    const handleTransaction = async () => {
        if (orderType === 'sell' && quantity > ownedQuantity) {
            setMessage({
                type: 'error',
                text: `You can only sell up to ${ownedQuantity} shares`
            });
            return;
        }

        const exactTotalCost = roundMoney(currentPrice * quantity);
        if (orderType === 'buy' && exactTotalCost > userBalance) {
            setMessage({
                type: 'error',
                text: 'Insufficient funds for this transaction'
            });
            return;
        }

        setIsLoading(true);
        setMessage(null);

        try {
            const token = localStorage.getItem('token');
            if (!token) throw new Error('No authentication token found');

            const response = await fetch('http://localhost:5000/api/transaction', {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol,
                    quantity,
                    type: orderType,
                    price: roundMoney(currentPrice)
                })
            });

            const data = await response.json();

            if (response.ok) {
                setMessage({
                    type: 'success',
                    text: `Successfully ${orderType === 'buy' ? 'bought' : 'sold'} ${quantity} shares of ${symbol}`
                });

                const balanceResponse = await fetch('http://localhost:5000/api/user/balance', {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Accept': 'application/json'
                    }
                });

                if (balanceResponse.ok) {
                    const balanceData = await balanceResponse.json();
                    setUserBalance(roundMoney(balanceData.balance));
                    setOwnedQuantity(prev => 
                        orderType === 'buy' ? prev + quantity : prev - quantity
                    );
                }

                onTransactionComplete?.();
            } else {
                throw new Error(data.detail || 'Transaction failed');
            }
        } catch (error) {
            setMessage({
                type: 'error',
                text: error instanceof Error ? error.message : 'Transaction failed'
            });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="bg-gray-50 rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold text-gray-900">Trade {symbol}</h3>
                <div className="text-right">
                    <div className="text-sm font-medium text-gray-600">Available Balance</div>
                    <div className="text-lg font-bold text-gray-900">
                        {isBalanceLoading ? (
                            <span className="text-gray-400">Loading...</span>
                        ) : (
                            `$${userBalance.toFixed(2)}`
                        )}
                    </div>
                </div>
            </div>

            <div className="space-y-6">
                <div className="flex space-x-4">
                    <button
                        onClick={() => setOrderType('buy')}
                        className={`flex-1 py-2 px-4 rounded-lg font-semibold text-base transition-colors ${
                            orderType === 'buy'
                                ? 'bg-green-600 text-white hover:bg-green-700'
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                    >
                        Buy
                    </button>
                    <button
                        onClick={() => setOrderType('sell')}
                        className={`flex-1 py-2 px-4 rounded-lg font-semibold text-base transition-colors ${
                            orderType === 'sell'
                                ? 'bg-red-600 text-white hover:bg-red-700'
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                    >
                        Sell
                    </button>
                </div>

                {orderType === 'sell' && (
                    <div className="text-sm text-gray-600">
                        Available to sell: {ownedQuantity} shares
                    </div>
                )}

                <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Quantity
                    </label>
                    <input
                        type="number"
                        min="1"
                        max={orderType === 'sell' ? ownedQuantity : undefined}
                        value={quantity}
                        onChange={handleQuantityChange}
                        className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                    />
                </div>

                <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <div className="flex justify-between mb-2">
                        <span className="text-gray-600 font-medium">Market Price</span>
                        <span className="text-gray-900 font-bold">
                            ${currentPrice.toFixed(2)}
                        </span>
                    </div>
                    <div className="flex justify-between mb-2">
                        <span className="text-gray-600 font-medium">Quantity</span>
                        <span className="text-gray-900 font-bold">{quantity}</span>
                    </div>
                    <div className="border-t border-gray-200 pt-2 mt-2">
                        <div className="flex justify-between">
                            <span className="text-gray-600 font-semibold">Estimated Total</span>
                            <span className="text-gray-900 font-bold">
                                ${totalCost.toFixed(2)}
                            </span>
                        </div>
                    </div>
                </div>

                {message && (
                    <Alert variant={message.type === 'success' ? 'default' : 'destructive'}>
                        <AlertDescription>{message.text}</AlertDescription>
                    </Alert>
                )}

                <button
                    onClick={handleTransaction}
                    disabled={
                        isLoading || 
                        (orderType === 'buy' && totalCost > userBalance) ||
                        (orderType === 'sell' && quantity > ownedQuantity)
                    }
                    className={`w-full py-3 px-4 rounded-lg font-semibold text-base transition-colors
                        ${orderType === 'buy' 
                            ? 'bg-green-600 text-white hover:bg-green-700' 
                            : 'bg-red-600 text-white hover:bg-red-700'}
                        ${(isLoading || 
                           (orderType === 'buy' && totalCost > userBalance) ||
                           (orderType === 'sell' && quantity > ownedQuantity)) 
                          ? 'opacity-50 cursor-not-allowed' 
                          : ''}`}
                >
                    {isLoading 
                        ? 'Processing...' 
                        : `${orderType === 'buy' ? 'Buy' : 'Sell'} ${quantity} ${symbol} Share${quantity > 1 ? 's' : ''}`}
                </button>
            </div>
        </div>
    );
};

export default BuySell;
