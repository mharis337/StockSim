"use client";

import Portfolio from '@/components/Portfolio';

export default function Dashboard() {
  return (
    <div className="h-full p-6">
      <div className="max-w-7xl mx-auto">
        <Portfolio />
      </div>
    </div>
  );
}