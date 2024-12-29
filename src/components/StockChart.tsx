"use client";

import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

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
}

const StockChart: React.FC<StockChartProps> = ({ data }) => {
  const chartRef = useRef<SVGSVGElement | null>(null);
  const [chartType, setChartType] = useState<"line" | "ticker">("line");

  useEffect(() => {
    if (!data || !Array.isArray(data) || data.length === 0 || !chartRef.current) return;

    const svg = d3.select(chartRef.current);
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };

    svg.selectAll("*").remove(); // Clear previous content

    // Filter data to include only trading hours (9:30 AM - 4:00 PM)
    const filteredData = data.filter((d) => {
      const date = new Date(d.Date);
      const hours = date.getHours();
      const minutes = date.getMinutes();
      return (hours === 9 && minutes >= 30) || (hours > 9 && hours < 16);
    });

    // X Scale
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(filteredData, (d) => new Date(d.Date)) as [Date, Date])
      .range([margin.left, width - margin.right]);

    // Y Scale
    const yScale = d3
      .scaleLinear()
      .domain([
        d3.min(filteredData, (d) => d.Low) || 0,
        d3.max(filteredData, (d) => d.High) || 0,
      ])
      .nice()
      .range([height - margin.bottom, margin.top]);

    const chartContent = svg.append("g").attr("clip-path", "url(#clip)");

    const xAxisGroup = svg
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height - margin.bottom})`);

    const yAxisGroup = svg
      .append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`);

    const updateAxes = (newXScale: any) => {
      const tickFormat = d3.timeFormat("%b %d %H:%M"); // Day:Time format

      xAxisGroup
        .call(d3.axisBottom(newXScale).ticks(10).tickFormat(tickFormat))
        .selectAll("text")
        .attr("fill", "#333")
        .style("font-size", "12px")
        .style("text-anchor", "end")
        .attr("transform", "rotate(-45)");

      xAxisGroup
        .select(".domain")
        .attr("stroke", "#666")
        .attr("stroke-width", 1.5);

      yAxisGroup
        .call(d3.axisLeft(yScale).ticks(10))
        .selectAll("text")
        .attr("fill", "#333")
        .style("font-size", "12px");
    };

    const renderChart = (newXScale: any) => {
      const visibleData = filteredData.filter((d) => {
        const date = new Date(d.Date);
        return date >= newXScale.domain()[0] && date <= newXScale.domain()[1];
      });

      chartContent.selectAll("*").remove(); // Clear previous content

      if (chartType === "line") {
        const line = d3
          .line<StockRow>()
          .x((d) => newXScale(new Date(d.Date)))
          .y((d) => yScale(d.Close))
          .curve(d3.curveMonotoneX);

        chartContent
          .append("path")
          .datum(visibleData)
          .attr("fill", "none")
          .attr("stroke", "#4F46E5")
          .attr("stroke-width", 2)
          .attr("d", line);
      } else if (chartType === "ticker") {
        chartContent
          .selectAll(".candle")
          .data(visibleData)
          .join("g")
          .attr("class", "candle")
          .each(function (d) {
            const group = d3.select(this);
            group.append("line")
              .attr("x1", newXScale(new Date(d.Date)))
              .attr("x2", newXScale(new Date(d.Date)))
              .attr("y1", yScale(d.High))
              .attr("y2", yScale(d.Low))
              .attr("stroke", "#333");

            group.append("rect")
              .attr("x", newXScale(new Date(d.Date)) - 2)
              .attr("y", yScale(Math.max(d.Open, d.Close)))
              .attr("width", 4)
              .attr("height", Math.abs(yScale(d.Open) - yScale(d.Close)))
              .attr("fill", d.Open > d.Close ? "red" : "green");
          });
      }
    };

    const zoom = d3
      .zoom()
      .scaleExtent([1, 90]) // Zoom scale range (1 minute to 90 minutes)
      .translateExtent([
        [margin.left, margin.top],
        [width - margin.right, height - margin.bottom],
      ])
      .on("zoom", (event) => {
        const { transform } = event;

        // Rescale the x-axis
        const zoomedXScale = transform.rescaleX(xScale);

        // Dynamically adjust intervals based on zoom level
        const newInterval =
          transform.k < 10
            ? 90
            : transform.k < 30
            ? 30
            : transform.k < 60
            ? 15
            : transform.k < 90
            ? 5
            : 1; // Decrease interval as zoom increases

        updateAxes(zoomedXScale);

        renderChart(zoomedXScale);
      });

    svg.call(zoom);

    // Initial render
    updateAxes(xScale);
    renderChart(xScale);
  }, [data, chartType]);

  return (
    <div style={{ textAlign: "center", marginBottom: "20px" }}>
      <div style={{ marginBottom: "10px" }}>
        <button
          onClick={() => setChartType("line")}
          style={{
            backgroundColor: chartType === "line" ? "#4F46E5" : "#E5E7EB",
            color: chartType === "line" ? "#FFFFFF" : "#333",
            padding: "10px 20px",
            border: "none",
            borderRadius: "5px",
            marginRight: "10px",
            fontSize: "14px",
            cursor: "pointer",
          }}
        >
          Line Chart
        </button>
        <button
          onClick={() => setChartType("ticker")}
          style={{
            backgroundColor: chartType === "ticker" ? "#4F46E5" : "#E5E7EB",
            color: chartType === "ticker" ? "#FFFFFF" : "#333",
            padding: "10px 20px",
            border: "none",
            borderRadius: "5px",
            fontSize: "14px",
            cursor: "pointer",
          }}
        >
          Ticker Chart
        </button>
      </div>
      <svg ref={chartRef} width="800" height="400"></svg>
    </div>
  );
};

export default StockChart;
