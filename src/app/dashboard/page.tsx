"use client";

import Portfolio from '@/components/Portfolio';
import Watchlist from '@/components/Watchlist';

export default function Dashboard() {
  return (
    <div className="min-h-screen p-8 bg-gradient-to-b from-blue-50 to-blue-100">
      <div className="max-w-[1400px] mx-auto">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          {/* Portfolio Section */}
          <div className="flex flex-col gap-8">
            <Portfolio />
          </div>
          
          {/* Watchlist Section */}
          <div >
            <Watchlist />
          </div>
        </div>
      </div>
    </div>
  );
}