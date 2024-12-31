import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import BuySell from './BuySell';

interface TradingModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  stockData: any[];
  initialAction?: 'buy' | 'sell';
}

const TradingModal: React.FC<TradingModalProps> = ({ 
  isOpen, 
  onClose, 
  symbol, 
  stockData, 
  initialAction = 'buy' 
}) => {
  const [ownedQuantity, setOwnedQuantity] = useState<number>(0);

  useEffect(() => {
    const fetchOwnedQuantity = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) return;

        const response = await fetch('http://localhost:5000/api/portfolio', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Accept': 'application/json'
          }
        });

        if (!response.ok) throw new Error('Failed to fetch portfolio');

        const data = await response.json();
        const holding = data.portfolio.find(h => h.symbol === symbol);
        setOwnedQuantity(holding ? holding.quantity : 0);
      } catch (error) {
        console.error('Error fetching owned quantity:', error);
      }
    };

    if (isOpen && symbol) {
      fetchOwnedQuantity();
    }
  }, [isOpen, symbol]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-xl max-h-[90vh] overflow-y-auto relative">
        <div className="sticky top-0 bg-white px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-900">
            Trade {symbol}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>
        
        <div className="p-6">
          <BuySell 
            stockData={stockData} 
            symbol={symbol} 
            initialAction={initialAction}
            maxSellQuantity={ownedQuantity}
          />
        </div>
      </div>
    </div>
  );
};

export default TradingModal;