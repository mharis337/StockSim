import React from 'react';
import { X } from 'lucide-react';
import BuySell from './BuySell';

interface TradingModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  stockData: any[];
  initialAction?: 'buy' | 'sell';
  maxSellQuantity?: number;
  onTransactionComplete?: () => void;  // Add callback for transaction completion
}

const TradingModal = ({ 
  isOpen, 
  onClose, 
  symbol, 
  stockData, 
  initialAction = 'buy', 
  maxSellQuantity,
  onTransactionComplete 
}: TradingModalProps) => {
  if (!isOpen) return null;

  const handleTransactionComplete = () => {
    onTransactionComplete?.();  // Call the callback if provided
    onClose();  // Close the modal
  };

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
            maxSellQuantity={maxSellQuantity}
            onTransactionComplete={handleTransactionComplete}  // Pass the callback
          />
        </div>
      </div>
    </div>
  );
};

export default TradingModal;