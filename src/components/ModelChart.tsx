import React, { useEffect } from 'react';
import * as d3 from 'd3';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter
} from 'recharts';

interface StockChartProps {
  data: any[];
  selectedIndicators: string[];
  xDomain: [number, number] | null;
  chartContainerRef: React.RefObject<HTMLDivElement>;
  onZoom: (newDomain: [number, number]) => void;
}

const COLORS = [
  '#4F46E5', '#EF4444', '#22C55E', '#F59E0B',
  '#3B82F6', '#8B5CF6', '#EC4899', '#F97316',
  '#14B8A6', '#6B7280',
];

export const StockChart: React.FC<StockChartProps> = ({
  data,
  selectedIndicators,
  xDomain,
  chartContainerRef,
  onZoom
}) => {
  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    setTimeout(() => {
      const svg = d3.select(chartContainerRef.current).select('svg');
      if (!svg.empty()) {
        const width = parseInt(svg.style('width')) || 800;
        const height = parseInt(svg.style('height')) || 400;
        const margin = { top: 20, right: 60, bottom: 70, left: 60 };

        const dataMin = Math.min(...data.map(d => d.timestamp));
        const dataMax = Math.max(...data.map(d => d.timestamp));

        const xScale = d3.scaleTime()
          .domain([new Date(dataMin), new Date(dataMax)])
          .range([margin.left, width - margin.right]);

        const zoom = d3.zoom<SVGSVGElement, unknown>()
          .scaleExtent([1, 20])
          .extent([[margin.left, 0], [width - margin.right, height]])
          .translateExtent([[margin.left, -Infinity], [width - margin.right, Infinity]])
          .on('zoom', (event) => {
            const newXScale = event.transform.rescaleX(xScale);
            const newDomain: [number, number] = [
              newXScale.invert(margin.left).getTime(),
              newXScale.invert(width - margin.right).getTime()
            ];

            // Ensure we don't zoom beyond data boundaries
            const clampedDomain: [number, number] = [
              Math.max(newDomain[0], dataMin),
              Math.min(newDomain[1], dataMax)
            ];

            onZoom(clampedDomain);
          });

        svg.call(zoom as any);
      }
    }, 100);

    return () => {
      const svg = d3.select(chartContainerRef.current).select('svg');
      if (!svg.empty()) {
        svg.on('.zoom', null);
      }
    };
  }, [data, onZoom, chartContainerRef]);

  if (selectedIndicators.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center text-gray-500">
        Please select at least one indicator to display.
      </div>
    );
  }

  const filteredData = xDomain
    ? data.filter(d => d.timestamp >= xDomain[0] && d.timestamp <= xDomain[1])
    : data;

  const yValues = selectedIndicators.flatMap(indicator =>
    filteredData
      .map(d => d[indicator])
      .filter(v => v !== undefined && v !== null && !isNaN(v))
  );

  const yDomain = yValues.length > 0
    ? [Math.min(...yValues) * 0.95, Math.max(...yValues) * 1.05]
    : ['auto', 'auto'];

  return (
    <div ref={chartContainerRef} className="relative">
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={filteredData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="timestamp"
            domain={xDomain || ['auto', 'auto']}
            tickFormatter={(timestamp) => {
              const date = new Date(timestamp);
              return date.toLocaleDateString() + ' ' + 
                     date.toLocaleTimeString([], { 
                       hour: '2-digit', 
                       minute: '2-digit' 
                     });
            }}
            scale="time"
          />
          <YAxis
            yAxisId="left"
            domain={yDomain}
            tickFormatter={(value: number) => value?.toFixed(2)}
            label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            labelFormatter={(label: number) => new Date(label).toLocaleString()}
            formatter={(value: any, name: string) => {
              if (value === null || value === undefined) return ['N/A', name];
              return [value.toFixed(2), name.replace(/_/g, ' ')];
            }}
          />
          <Legend />

          {selectedIndicators.map((indicator, index) => (
            <Line
              key={indicator}
              type="monotone"
              dataKey={indicator}
              stroke={COLORS[index % COLORS.length]}
              dot={false}
              yAxisId="left"
              connectNulls
              name={indicator.replace(/_/g, ' ')}
              isAnimationActive={false}
            />
          ))}

          <Scatter
            name="Buy Signal"
            data={filteredData.filter(d => d.buySignal)}
            fill="#22C55E"
            shape="triangle"
            yAxisId="left"
          />
          <Scatter
            name="Sell Signal"
            data={filteredData.filter(d => d.sellSignal)}
            fill="#EF4444"
            shape="triangle"
            yAxisId="left"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="text-sm text-gray-600 text-center mt-2">
        Use mouse wheel to zoom, drag to pan
      </div>
    </div>
  );
};