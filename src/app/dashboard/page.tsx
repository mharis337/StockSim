// src/app/dashboard/page.tsx
"use client";

import Portfolio from '@/components/Portfolio';

export default function Dashboard() {
  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <Portfolio />
    </div>
  );
}