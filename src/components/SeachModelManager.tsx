import React from 'react';

interface SearchBarProps {
  searchSymbol: string;
  setSearchSymbol: (symbol: string) => void;
  handleSearch: () => void;
}

export const SearchBar: React.FC<SearchBarProps> = ({ 
  searchSymbol, 
  setSearchSymbol, 
  handleSearch 
}) => (
  <div className="flex items-center space-x-2 w-full md:w-auto">
    <input
      type="text"
      value={searchSymbol}
      onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
      className="flex-1 p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
      placeholder="Enter stock symbol (e.g., AAPL)"
      aria-label="Stock Symbol"
    />
    <button
      onClick={handleSearch}
      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
    >
      Search
    </button>
  </div>
);