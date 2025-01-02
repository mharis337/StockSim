import React from 'react';

interface IndicatorSelectorProps {
  indicatorGroups: { [key: string]: string[] };
  selectedIndicators: string[];
  setSelectedIndicators: (indicators: string[]) => void;
  dropdownOpen: { [key: string]: boolean };
  setDropdownOpen: (state: { [key: string]: boolean }) => void;
}

export const IndicatorSelector: React.FC<IndicatorSelectorProps> = ({
  indicatorGroups,
  selectedIndicators,
  setSelectedIndicators,
  dropdownOpen,
  setDropdownOpen,
}) => (
  <div className="mb-4 flex flex-wrap gap-4">
    {Object.entries(indicatorGroups).map(([group, indicators]) => (
      <div key={group} className="relative">
        <button
          onClick={() => setDropdownOpen(prev => ({ ...prev, [group]: !prev[group] }))}
          className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center inline-flex items-center"
          type="button"
        >
          {group}
          <svg className="w-2.5 h-2.5 ml-3" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 10 6">
            <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m1 1 4 4 4-4" />
          </svg>
        </button>

        {dropdownOpen[group] && (
          <div className="absolute z-10 bg-white divide-y divide-gray-100 rounded-lg shadow w-56 mt-2">
            <ul className="py-2 text-sm text-gray-700">
              {indicators.map((indicator) => (
                <li key={indicator}>
                  <label className="flex items-center px-4 py-2 hover:bg-gray-100 cursor-pointer">
                    <input
                      type="checkbox"
                      className="form-checkbox h-4 w-4 text-blue-600"
                      checked={selectedIndicators.includes(indicator)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedIndicators([...selectedIndicators, indicator]);
                        } else {
                          setSelectedIndicators(selectedIndicators.filter(i => i !== indicator));
                        }
                      }}
                    />
                    <span className="ml-2">{indicator.replace(/_/g, ' ')}</span>
                  </label>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    ))}
  </div>
);