"use client";

import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

interface StockRow {
  Date: string; 
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
}

interface StockChartProps {
  data: StockRow[];
  interval: string;  
}

const StockChart: React.FC<StockChartProps> = ({ data, interval }) => {
  const chartRef = useRef<HTMLCanvasElement>(null);

  const intradayIntervals = ["1m","2m","5m","15m","30m","60m","90m","1h"];
  const isIntraday = intradayIntervals.includes(interval);
  
  const transformedData = data.map((row) => {
    const dateObj = new Date(row.Date);
    console.log(dateObj);
  
    const xLabel = isIntraday
      ? dateObj.toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          timeZone: "America/New_York",
        })
      : dateObj.toLocaleDateString("en-US", { timeZone: "America/New_York" });
  
    return {
      xLabel,
      close: row.Close,
    };
  });

  useEffect(() => {
    if (!transformedData || transformedData.length === 0) return;
    const labels = transformedData.map((d) => d.xLabel);
    const values = transformedData.map((d) => d.close);

    const chart = new Chart(chartRef.current!, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Closing Price",
            data: values,
            borderColor: "#4F46E5",
            backgroundColor: "rgba(79, 70, 229, 0.2)",
            borderWidth: 2,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 10,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: true,
            labels: {
              color: "#333",
              font: { size: 14, family: "Inter, sans-serif" },
            },
          },
          tooltip: {
            backgroundColor: "#1F2937",
            titleColor: "#fff",
            bodyColor: "#fff",
          },
        },
        scales: {
          x: {
            ticks: {
              color: "#333",
              font: { size: 12 },
            },
            grid: { display: false },
          },
          y: {
            ticks: {
              color: "#333",
              font: { size: 12 },
            },
            grid: { color: "rgba(0, 0, 0, 0.1)" },
          },
        },
      },
    });
    
    

    return () => {
      chart.destroy();
    };
  }, [transformedData]);

  return <canvas ref={chartRef}></canvas>;
};

export default StockChart;
