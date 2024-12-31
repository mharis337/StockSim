import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
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

      const data = await response.json();
      if (data.portfolio && Array.isArray(data.portfolio)) {
        const validHoldings = data.portfolio.filter(holding => holding.quantity > 0);
        setPortfolio(validHoldings);
      } else {
        setPortfolio([]);
      }
    } catch (err) {
      console.error('Portfolio fetch error:', err);
      setError(err.message);
      setPortfolio([]); // Ensure portfolio is set to an empty array on error
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
          'Content-Type': 'application/json',
        },
      });
  
      if (!response.ok) throw new Error('Failed to fetch transaction history');
  
      const data = await response.json();
      console.log('Fetched transaction data:', data);
  
      if (data.recentTransactions && Array.isArray(data.recentTransactions)) {
        setTransactions(data.recentTransactions); // Use the correct key here
      } else {
        console.warn('Invalid transactions data received:', data.recentTransactions);
        setTransactions([]); // Fallback to empty array
      }
    } catch (err) {
      console.error('Error fetching transactions:', err);
      setTransactions([]); // Optionally reset to empty array on error
    }
  };

  useEffect(() => {
    fetchPortfolio();
    fetchTransactionHistory();
    const intervalId = setInterval(fetchPortfolio, 60000);
    return () => clearInterval(intervalId);
  }, []);

  const totalValue = portfolio.reduce((sum, holding) => sum + holding.market_value, 0);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
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
    <div className="space-y-6 p-6">
      <div className="bg-white/50 backdrop-blur-sm rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">Your Portfolio</h2>
          <div className="text-right">
            <div className="text-sm font-medium text-gray-700">Total Value</div>
            <div className="text-2xl font-bold text-gray-800">
              ${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-gray-700">Symbol</th>
                <th className="text-right py-3 px-4 text-gray-700">Shares</th>
                <th className="text-right py-3 px-4 text-gray-700">Current Price</th>
                <th className="text-right py-3 px-4 text-gray-700">Market Value</th>
              </tr>
            </thead>
            <tbody>
              {portfolio.length === 0 ? (
                <tr>
                  <td colSpan={4} className="text-center py-4 text-gray-600">
                    No holdings found in your portfolio
                  </td>
                </tr>
              ) : (
                portfolio.map((holding) => (
                  <tr key={holding.symbol} className="border-b border-gray-200 hover:bg-white/30">
                    <td className="py-3 px-4 font-medium text-gray-800">{holding.symbol}</td>
                    <td className="text-right py-3 px-4 text-gray-700">{holding.quantity}</td>
                    <td className="text-right py-3 px-4 text-gray-700">
                      {holding.price_unavailable ? (
                        <span className="text-gray-500">Updating...</span>
                      ) : (
                        `$${holding.current_price.toFixed(2)}`
                      )}
                    </td>
                    <td className="text-right py-3 px-4 font-medium text-gray-800">
                      {holding.price_unavailable ? (
                        <span className="text-gray-500">Updating...</span>
                      ) : (
                        `$${holding.market_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white/50 backdrop-blur-sm rounded-lg shadow-lg">
        <div 
          className="p-4 flex justify-between items-center cursor-pointer hover:bg-white/30"
          onClick={() => setShowTransactions(!showTransactions)}
        >
          <h3 className="text-lg font-semibold text-gray-800">Transaction History</h3>
          <button className="text-blue-600 hover:text-blue-700 font-medium">
            {showTransactions ? 'Hide' : 'Show'}
          </button>
        </div>

        {showTransactions && (
          <div className="p-4 border-t border-gray-200">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-4 text-gray-700">Date</th>
                    <th className="text-left py-2 px-4 text-gray-700">Type</th>
                    <th className="text-left py-2 px-4 text-gray-700">Symbol</th>
                    <th className="text-right py-2 px-4 text-gray-700">Quantity</th>
                    <th className="text-right py-2 px-4 text-gray-700">Price</th>
                    <th className="text-right py-2 px-4 text-gray-700">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.isArray(transactions) && transactions.length > 0 ? (
                    transactions.map((tx, index) => (
                      <tr key={index} className="border-b border-gray-100 hover:bg-white/30">
                        <td className="py-2 px-4 text-gray-700">
                          {new Date(tx.timestamp).toLocaleDateString()}
                        </td>
                        <td className="py-2 px-4">
                          <span className={`flex items-center ${
                            tx.type === 'buy' ? 'text-green-600' : 'text-red-600'
                          } font-medium`}>
                            {tx.type === 'buy' ? (
                              <ArrowUpCircle className="w-4 h-4 mr-1" />
                            ) : (
                              <ArrowDownCircle className="w-4 h-4 mr-1" />
                            )}
                            {tx.type.charAt(0).toUpperCase() + tx.type.slice(1)}
                          </span>
                        </td>
                        <td className="py-2 px-4 text-gray-700">{tx.symbol}</td>
                        <td className="text-right py-2 px-4 text-gray-700">{tx.quantity}</td>
                        <td className="text-right py-2 px-4 text-gray-700">${tx.price.toFixed(2)}</td>
                        <td className="text-right py-2 px-4 font-medium text-gray-800">
                          ${(tx.quantity * tx.price).toFixed(2)}
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="text-center py-4 text-gray-600">
                        No transactions found
                      </td>
                    </tr>
                  )}
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
