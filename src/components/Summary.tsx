import React, { useState, useEffect } from 'react';
import { ArrowUpCircle, ArrowDownCircle, DollarSign, Briefcase, PiggyBank, TrendingUp } from 'lucide-react';

const PortfolioSummary = () => {
  const [portfolioData, setPortfolioData] = useState({
    totalBalance: 0,
    cashBalance: 0,
    equity: 0,
    dayPL: 0,
    dayPLPercent: 0,
    totalPL: 0,
    totalPLPercent: 0
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchPortfolioData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('No authentication token found');
        }

        const response = await fetch('http://localhost:5000/api/portfolio/history', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Accept': 'application/json'
          }
        });

        if (!response.ok) {
          throw new Error('Failed to fetch portfolio data');
        }

        const data = await response.json();
        
        // Ensure we have valid numbers or default to 0
        const cashBalance = Number(data.cashBalance) || 0;
        const equity = Number(data.equity) || 0;
        const totalValue = Number(data.totalValue) || (cashBalance + equity);
        
        // Calculate P/L values
        const todaysPL = equity * 0.02; // Simplified daily P/L calculation
        const totalPL = equity - 1000; // Assuming initial investment was 1000
        
        // Calculate percentages safely
        const dayPLPercent = totalValue !== 0 ? (todaysPL / totalValue) * 100 : 0;
        const totalPLPercent = totalPL !== 0 ? (totalPL / 1000) * 100 : 0;

        setPortfolioData({
          totalBalance: totalValue,
          cashBalance: cashBalance,
          equity: equity,
          dayPL: todaysPL,
          dayPLPercent: dayPLPercent,
          totalPL: totalPL,
          totalPLPercent: totalPLPercent
        });

      } catch (error) {
        console.error('Error fetching portfolio data:', error);
        setError('Failed to load portfolio data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchPortfolioData();
    
    // Refresh data every minute
    const intervalId = setInterval(fetchPortfolioData, 60000);
    return () => clearInterval(intervalId);
  }, []);

  // Format currency value with proper handling of invalid numbers
  const formatCurrency = (value) => {
    if (typeof value !== 'number' || isNaN(value)) {
      return '$0.00';
    }
    return value.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  // Format percentage with proper handling of invalid numbers
  const formatPercentage = (value) => {
    if (typeof value !== 'number' || isNaN(value)) {
      return '0.00%';
    }
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="text-red-600 bg-red-50 p-4 rounded-lg">
          {error}
        </div>
      </div>
    );
  }

  const metrics = [
    {
      title: 'Total Portfolio Value',
      value: portfolioData.totalBalance,
      icon: Briefcase,
      format: 'currency'
    },
    {
      title: 'Cash Balance',
      value: portfolioData.cashBalance,
      icon: DollarSign,
      format: 'currency'
    },
    {
      title: 'Equity Value',
      value: portfolioData.equity,
      icon: PiggyBank,
      format: 'currency'
    },
    {
      title: "Today's P/L",
      value: portfolioData.dayPL,
      percentage: portfolioData.dayPLPercent,
      icon: portfolioData.dayPL >= 0 ? ArrowUpCircle : ArrowDownCircle,
      format: 'pl'
    },
    {
      title: 'Total P/L',
      value: portfolioData.totalPL,
      percentage: portfolioData.totalPLPercent,
      icon: portfolioData.totalPL >= 0 ? TrendingUp : ArrowDownCircle,
      format: 'pl'
    }
  ];

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-6">Portfolio Summary</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((metric, index) => (
          <div
            key={index}
            className="p-4 rounded-xl bg-gradient-to-br from-white to-gray-50 border border-gray-100 hover:border-gray-200 transition-all duration-200 shadow-sm hover:shadow"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600">{metric.title}</span>
              <metric.icon
                className={`w-5 h-5 ${
                  metric.format === 'pl'
                    ? metric.value >= 0
                      ? 'text-green-500'
                      : 'text-red-500'
                    : 'text-blue-500'
                }`}
              />
            </div>
            <div className="space-y-1">
              <div className="text-2xl font-bold text-gray-900">
                {formatCurrency(metric.value)}
              </div>
              {metric.percentage !== undefined && (
                <div
                  className={`text-sm font-medium ${
                    metric.value >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {formatPercentage(metric.percentage)}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PortfolioSummary;