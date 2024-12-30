import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ArrowUpCircle, ArrowDownCircle } from 'lucide-react';

const Portfolio = () => {
  const router = useRouter();
  const [portfolio, setPortfolio] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [showTransactions, setShowTransactions] = useState(false);

  const fetchPortfolio = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      const response = await fetch('http://localhost:5000/api/portfolio', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.status === 401) {
        router.push('/login');
        return;
      }

      if (!response.ok) {
        throw new Error('Failed to fetch portfolio');
      }

      const data = await response.json();
      setPortfolio(data.portfolio);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchTransactionHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch('http://localhost:5000/api/portfolio/history', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) throw new Error('Failed to fetch transaction history');

      const data = await response.json();
      setTransactions(data.transactions);
    } catch (err) {
      console.error('Error fetching transactions:', err);
    }
  };

  useEffect(() => {
    fetchPortfolio();
    fetchTransactionHistory();
  }, []);

  const totalValue = portfolio.reduce((sum, holding) => sum + holding.market_value, 0);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Your Portfolio</h2>
          <div className="text-right">
            <div className="text-sm text-gray-500">Total Value</div>
            <div className="text-2xl font-bold text-gray-900">
              ${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </div>
          </div>
        </div>

        {/* Holdings Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4">Symbol</th>
                <th className="text-right py-3 px-4">Shares</th>
                <th className="text-right py-3 px-4">Current Price</th>
                <th className="text-right py-3 px-4">Market Value</th>
              </tr>
            </thead>
            <tbody>
              {portfolio.map((holding) => (
                <tr key={holding.symbol} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4 font-medium">{holding.symbol}</td>
                  <td className="text-right py-3 px-4">{holding.quantity}</td>
                  <td className="text-right py-3 px-4">
                    ${holding.current_price.toFixed(2)}
                  </td>
                  <td className="text-right py-3 px-4 font-medium">
                    ${holding.market_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Transaction History */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex justify-between items-center cursor-pointer hover:bg-gray-50"
          onClick={() => setShowTransactions(!showTransactions)}
        >
          <h3 className="text-lg font-semibold text-gray-900">Transaction History</h3>
          <button className="text-blue-600 hover:text-blue-700">
            {showTransactions ? 'Hide' : 'Show'}
          </button>
        </div>

        {showTransactions && (
          <div className="p-4 border-t border-gray-100">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-4">Date</th>
                    <th className="text-left py-2 px-4">Type</th>
                    <th className="text-left py-2 px-4">Symbol</th>
                    <th className="text-right py-2 px-4">Quantity</th>
                    <th className="text-right py-2 px-4">Price</th>
                    <th className="text-right py-2 px-4">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.map((tx, index) => (
                    <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-2 px-4">
                        {new Date(tx.timestamp).toLocaleDateString()}
                      </td>
                      <td className="py-2 px-4">
                        <span className={`flex items-center ${
                          tx.type === 'buy' ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {tx.type === 'buy' ? (
                            <ArrowUpCircle className="w-4 h-4 mr-1" />
                          ) : (
                            <ArrowDownCircle className="w-4 h-4 mr-1" />
                          )}
                          {tx.type.charAt(0).toUpperCase() + tx.type.slice(1)}
                        </span>
                      </td>
                      <td className="py-2 px-4">{tx.symbol}</td>
                      <td className="text-right py-2 px-4">{tx.quantity}</td>
                      <td className="text-right py-2 px-4">${tx.price.toFixed(2)}</td>
                      <td className="text-right py-2 px-4 font-medium">
                        ${(tx.quantity * tx.price).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Portfolio;