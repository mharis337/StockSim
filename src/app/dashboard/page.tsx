"use client";

import Portfolio from '@/components/Portfolio';
import Watchlist from '@/components/Watchlist';
import PortfolioSummary from '@/components/Summary';

export default function Dashboard() {
  return (
    <div className="min-h-screen p-8 bg-gradient-to-b from-blue-50 to-blue-100">
      <div className="max-w-[1400px] mx-auto">
        {/* Portfolio Summary - full width */}
        <div className="mb-8">
          <PortfolioSummary />
        </div>
        
        <div className="flex flex-col lg:flex-row gap-8 justify-center">
          {/* Portfolio Section - fixed width */}
          <div className="lg:w-[850px] lg:flex-none">
            <Portfolio />
          </div>
          
          {/* Watchlist Section - fixed width */}
          <div className="lg:w-[350px] lg:flex-none">
            <Watchlist />
          </div>
        </div>
      </div>
    </div>
  );
}