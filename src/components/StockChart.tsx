import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface StockRow {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  ContinuousTime?: number;
}

interface StockChartProps {
  data: StockRow[];
  interval: string;
}

const StockChart: React.FC<StockChartProps> = ({ data, interval }) => {
  const chartRef = useRef<SVGSVGElement | null>(null);
  const [chartType, setChartType] = useState<"line" | "ticker">("line");

  useEffect(() => {
    if (!data || !Array.isArray(data) || data.length === 0 || !chartRef.current) return;

    const svg = d3.select(chartRef.current);
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 60, bottom: 70, left: 60 };

    svg.selectAll("*").remove();

    // Create tooltip
    let tooltip = d3.select("body").select(".stock-tooltip");
    if (tooltip.empty()) {
      tooltip = d3.select("body")
        .append("div")
        .attr("class", "stock-tooltip")
        .style("opacity", 0)
        .style("position", "absolute")
        .style("background-color", "rgba(0, 0, 0, 0.8)")
        .style("color", "white")
        .style("padding", "12px")
        .style("border-radius", "6px")
        .style("border", "1px solid #E5E7EB")
        .style("font-size", "14px")
        .style("font-weight", "500")
        .style("pointer-events", "none")
        .style("z-index", "100")
        .style("box-shadow", "0 4px 6px -1px rgba(0, 0, 0, 0.1)");
    }

    // Process data
    const filteredData = data.filter((d) => {
      const date = new Date(d.Date);
      const hours = date.getHours();
      const minutes = date.getMinutes();
      return (hours === 9 && minutes >= 30) || (hours > 9 && hours < 16);
    });

    let cumulativeOffset = 0;
    const mappedData = filteredData.map((d, i, arr) => {
      const currentDate = new Date(d.Date);
      const prevDate = i > 0 ? new Date(arr[i - 1].Date) : null;

      if (prevDate && 
          (currentDate.getDate() !== prevDate.getDate() ||
           currentDate.getMonth() !== prevDate.getMonth() ||
           currentDate.getFullYear() !== prevDate.getFullYear())) {
        cumulativeOffset += 390;
      }

      const tradingStart = new Date(currentDate);
      tradingStart.setHours(9, 30, 0, 0);
      const minutesSinceStart = (currentDate.getTime() - tradingStart.getTime()) / (1000 * 60);

      return {
        ...d,
        ContinuousTime: cumulativeOffset + minutesSinceStart,
      };
    });

    // Set up scales
    const xDomain = d3.extent(mappedData, (d) => d.ContinuousTime) as [number, number];
    const yDomain = [
      d3.min(mappedData, (d) => Math.min(d.Low, d.Close)) ?? 0,
      d3.max(mappedData, (d) => Math.max(d.High, d.Close)) ?? 0,
    ];

    const xScaleBase = d3
      .scaleLinear()
      .domain(xDomain)
      .range([margin.left, width - margin.right]);

    const yScale = d3
      .scaleLinear()
      .domain(yDomain)
      .nice()
      .range([height - margin.bottom, margin.top]);

    // Create clip path
    svg.append("defs")
      .append("clipPath")
      .attr("id", "clip")
      .append("rect")
      .attr("x", margin.left)
      .attr("y", margin.top)
      .attr("width", width - margin.left - margin.right)
      .attr("height", height - margin.top - margin.bottom);

    // Create chart content group
    const chartContent = svg.append("g")
      .attr("clip-path", "url(#clip)");

    const formatTime = d3.timeFormat("%b %d, %Y %H:%M");

    function updateChart(xScale: d3.ScaleLinear<number, number>) {
      // Update x-axis
      svg.select(".x-axis")
        .call(d3.axisBottom(xScale)
          .ticks(10)
          .tickFormat((d) => {
            const minutes = Number(d);
            const dayOffset = Math.floor(minutes / 390);
            const intraDay = minutes % 390;
            const baseDate = new Date(mappedData[0].Date);
            baseDate.setDate(baseDate.getDate() + dayOffset);
            baseDate.setHours(9, 30, 0, 0);
            baseDate.setMinutes(baseDate.getMinutes() + intraDay);
            return d3.timeFormat("%b %d %H:%M")(baseDate);
          }) as any)
        .selectAll("text")
        .style("text-anchor", "end")
        .style("font-size", "12px")
        .style("font-weight", "500")
        .style("fill", "#4B5563")
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-65)");

      chartContent.selectAll("*").remove();

      if (chartType === "line") {
        const line = d3
          .line<StockRow>()
          .x((d) => xScale(d.ContinuousTime!))
          .y((d) => yScale(d.Close));

        // Add the line
        chartContent
          .append("path")
          .datum(mappedData)
          .attr("fill", "none")
          .attr("stroke", "#4F46E5")
          .attr("stroke-width", 2)
          .attr("d", line);

        // Add hover line and point
        const hoverLine = chartContent
          .append("line")
          .style("opacity", "0")
          .attr("stroke", "#9CA3AF")
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "3,3");

        const hoverPoint = chartContent
          .append("circle")
          .style("opacity", "0")
          .attr("r", 4)
          .attr("fill", "#4F46E5")
          .attr("stroke", "white")
          .attr("stroke-width", 2);

        // Add hover overlay
        chartContent
          .append("rect")
          .attr("class", "overlay")
          .attr("fill", "none")
          .attr("pointer-events", "all")
          .attr("x", margin.left)
          .attr("y", margin.top)
          .attr("width", width - margin.left - margin.right)
          .attr("height", height - margin.top - margin.bottom)
          .on("mousemove", function(event) {
            const [mouseX] = d3.pointer(event);
            const xValue = xScale.invert(mouseX);
            const bisect = d3.bisector((d: StockRow) => d.ContinuousTime!).left;
            const index = bisect(mappedData, xValue);
            const d0 = mappedData[index - 1];
            const d1 = mappedData[index];
            const d = xValue - d0?.ContinuousTime! > d1?.ContinuousTime! - xValue ? d1 : d0;

            if (d) {
              hoverLine
                .style("opacity", "1")
                .attr("x1", xScale(d.ContinuousTime!))
                .attr("x2", xScale(d.ContinuousTime!))
                .attr("y1", margin.top)
                .attr("y2", height - margin.bottom);

              hoverPoint
                .style("opacity", "1")
                .attr("cx", xScale(d.ContinuousTime!))
                .attr("cy", yScale(d.Close));

              tooltip
                .style("opacity", 1)
                .html(`
                  <div class="font-bold text-gray-900 mb-1">${formatTime(new Date(d.Date))}</div>
                  <div class="grid grid-cols-2 gap-x-4 text-sm">
                    <div class="text-gray-600">Price:</div>
                    <div class="text-gray-900 font-semibold">$${d.Close.toFixed(2)}</div>
                    <div class="text-gray-600">Volume:</div>
                    <div class="text-gray-900 font-semibold">${d.Volume.toLocaleString()}</div>
                  </div>
                `)
                .style("left", `${event.pageX + 15}px`)
                .style("top", `${event.pageY - 28}px`);
            }
          })
          .on("mouseleave", function() {
            hoverLine.style("opacity", "0");
            hoverPoint.style("opacity", "0");
            tooltip.style("opacity", 0);
          });

        // Add current price dot
        const lastPoint = mappedData[mappedData.length - 1];
        chartContent
          .append("circle")
          .attr("cx", xScale(lastPoint.ContinuousTime!))
          .attr("cy", yScale(lastPoint.Close))
          .attr("r", 4)
          .attr("fill", lastPoint.Close >= mappedData[mappedData.length - 2].Close ? "#22C55E" : "#EF4444")
          .attr("stroke", "white")
          .attr("stroke-width", 2);

        // Add current price label
        chartContent
          .append("text")
          .attr("x", xScale(lastPoint.ContinuousTime!) + 8)
          .attr("y", yScale(lastPoint.Close))
          .attr("dy", "0.3em")
          .style("font-size", "14px")
          .style("font-weight", "600")
          .attr("fill", lastPoint.Close >= mappedData[mappedData.length - 2].Close ? "#22C55E" : "#EF4444")
          .text(`$${lastPoint.Close.toFixed(2)}`);

      } else {
        chartContent
          .selectAll(".candle")
          .data(mappedData)
          .join("g")
          .attr("class", "candle")
          .each(function(d) {
            const g = d3.select(this);
            const color = d.Open > d.Close ? "#EF4444" : "#22C55E";

            g.append("line")
              .attr("x1", xScale(d.ContinuousTime!))
              .attr("x2", xScale(d.ContinuousTime!))
              .attr("y1", yScale(d.High))
              .attr("y2", yScale(d.Low))
              .attr("stroke", color);

            const bodyHeight = Math.max(1, Math.abs(yScale(d.Open) - yScale(d.Close)));
            g.append("rect")
              .attr("x", xScale(d.ContinuousTime!) - 4)
              .attr("y", yScale(Math.max(d.Open, d.Close)))
              .attr("width", 8)
              .attr("height", bodyHeight)
              .attr("fill", color)
              .on("mouseover", function(event) {
                tooltip
                  .style("opacity", 1)
                  .html(`
                    <div class="font-bold text-gray-900 mb-1">${formatTime(new Date(d.Date))}</div>
                    <div class="grid grid-cols-2 gap-x-4 text-sm">
                      <div class="text-gray-600">Open:</div>
                      <div class="text-gray-900 font-semibold">$${d.Open.toFixed(2)}</div>
                      <div class="text-gray-600">High:</div>
                      <div class="text-gray-900 font-semibold">$${d.High.toFixed(2)}</div>
                      <div class="text-gray-600">Low:</div>
                      <div class="text-gray-900 font-semibold">$${d.Low.toFixed(2)}</div>
                      <div class="text-gray-600">Close:</div>
                      <div class="text-gray-900 font-semibold">$${d.Close.toFixed(2)}</div>
                      <div class="text-gray-600">Volume:</div>
                      <div class="text-gray-900 font-semibold">${d.Volume.toLocaleString()}</div>
                    </div>
                  `)
                  .style("left", `${event.pageX + 15}px`)
                  .style("top", `${event.pageY - 28}px`);
              })
              .on("mouseout", function() {
                tooltip.style("opacity", 0);
              });
          });
      }
    }

    // Add x-axis
    svg.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height - margin.bottom})`);

    // Add y-axis
    svg.append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale)
        .tickFormat(d => `$${d.toFixed(2)}`))
      .selectAll("text")
      .style("font-size", "12px")
      .style("font-weight", "500")
      .style("fill", "#4B5563");

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 20])
      .extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]])
      .translateExtent([[margin.left, -Infinity], [width - margin.right, Infinity]])
      .on("zoom", (event) => {
        const newXScale = event.transform.rescaleX(xScaleBase);
        updateChart(newXScale);
      });

    svg.call(zoom as any);

    // Initial render
    updateChart(xScaleBase);

  }, [data, chartType, interval]);

  return (
    <div className="flex flex-col items-center gap-4 bg-white p-4 rounded-lg shadow-sm">
      <div className="flex gap-2">
        <button
          onClick={() => setChartType("line")}
          className={`px-4 py-2 rounded-md text-base font-semibold transition-colors ${
            chartType === "line"
              ? "bg-blue-600 text-white shadow-sm"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          Line Chart
        </button>
        <button
          onClick={() => setChartType("ticker")}
          className={`px-4 py-2 rounded-md text-base font-semibold transition-colors ${
            chartType === "ticker"
              ? "bg-blue-600 text-white shadow-sm"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          Candlestick
        </button>
      </div>
      
      <svg 
        ref={chartRef} 
        width="800" 
        height="400"
        className="bg-white"
      />
      
      <div className="text-sm text-gray-600 mt-2">
        Scroll to zoom, drag to pan
      </div>
    </div>
  );
};

export default StockChart;