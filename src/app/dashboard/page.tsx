// src/app/dashboard/page.tsx
"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

interface StockData {
  symbol: string;
  price: number;
  change: number;
}

export default function Dashboard() {
  const router = useRouter();
  const [watchlist, setWatchlist] = useState<StockData[]>([
    { symbol: "AAPL", price: 182.89, change: 0.5 },
    { symbol: "GOOGL", price: 141.76, change: -0.3 },
    { symbol: "MSFT", price: 338.47, change: 1.2 }
  ]);
  const [userEmail, setUserEmail] = useState<string>("");

  useEffect(() => {
    const email = localStorage.getItem("userEmail");
    if (email) setUserEmail(email);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("userEmail");
    router.push("/login");
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Main Content */}
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {/* Overview Cards */}
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
          <div className="bg-white overflow-hidden shadow rounded-lg p-6">
            <div className="text-gray-500 text-sm">Portfolio Value</div>
            <div className="text-2xl font-semibold text-gray-900">$100,000.00</div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg p-6">
            <div className="text-gray-500 text-sm">Today's Return</div>
            <div className="text-2xl font-semibold text-green-600">+$1,234.56</div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg p-6">
            <div className="text-gray-500 text-sm">Buying Power</div>
            <div className="text-2xl font-semibold text-gray-900">$25,000.00</div>
          </div>
        </div>

        {/* Watchlist */}
        <div className="mt-8 bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Watchlist</h2>
          <div className="space-y-4">
            {watchlist.map((stock) => (
              <div 
                key={stock.symbol} 
                className="flex justify-between items-center border-b pb-2"
              >
                <div className="font-medium">{stock.symbol}</div>
                <div className="flex items-center space-x-4">
                  <span className="text-gray-900">${stock.price.toFixed(2)}</span>
                  <span className={stock.change >= 0 ? 'text-green-600' : 'text-red-600'}>
                    {stock.change >= 0 ? '+' : ''}{stock.change}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}