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

    // ------------------------------------------------
    // 1) Basic chart setup
    // ------------------------------------------------
    const svg = d3.select(chartRef.current);
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 70, left: 50 };

    svg.selectAll("*").remove(); // Clear previous render

    // ------------------------------------------------
    // 2) Filter data to trading hours (9:30 - 16:00)
    // ------------------------------------------------
    const filteredData = data.filter((d) => {
      const date = new Date(d.Date);
      const hours = date.getHours();
      const minutes = date.getMinutes();
      return (hours === 9 && minutes >= 30) || (hours > 9 && hours < 16);
    });

    // ------------------------------------------------
    // 3) Compute ContinuousTime (skipping nights)
    // ------------------------------------------------
    let cumulativeOffset = 0;
    const mappedData = filteredData.map((d, i, arr) => {
      const currentDate = new Date(d.Date);
      const prevDate = i > 0 ? new Date(arr[i - 1].Date) : null;

      // If new trading day, add 390 minutes (6.5 hours)
      if (
        prevDate &&
        (currentDate.getDate() !== prevDate.getDate() ||
          currentDate.getMonth() !== prevDate.getMonth() ||
          currentDate.getFullYear() !== prevDate.getFullYear())
      ) {
        cumulativeOffset += 390;
      }

      // Offset from 9:30 a.m.
      const tradingStart = new Date(currentDate);
      tradingStart.setHours(9, 30, 0, 0);
      const minutesSinceStart =
        (currentDate.getTime() - tradingStart.getTime()) / (1000 * 60);

      return {
        ...d,
        ContinuousTime: cumulativeOffset + minutesSinceStart,
      };
    });

    // ------------------------------------------------
    // 4) Determine X / Y domains
    // ------------------------------------------------
    const xDomain = d3.extent(mappedData, (d) => d.ContinuousTime) as [number, number];
    const [domainStart, domainEnd] = xDomain;
    const chartSpan = domainEnd - domainStart; // total coverage in minutes

    const yDomain = [
      d3.min(mappedData, (d) => d.Low) ?? 0,
      d3.max(mappedData, (d) => d.High) ?? 0,
    ];

    // Base scales (before zooming)
    const xScaleBase = d3
      .scaleLinear()
      .domain([domainStart, domainEnd])
      .range([margin.left, width - margin.right]);

    const yScale = d3
      .scaleLinear()
      .domain(yDomain)
      .nice()
      .range([height - margin.bottom, margin.top]);

    // ------------------------------------------------
    // 5) Setup chart groups
    // ------------------------------------------------
    const chartContent = svg.append("g").attr("clip-path", "url(#clip)");

    const xAxisGroup = svg
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height - margin.bottom})`);

    const yAxisGroup = svg
      .append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`);

    // ------------------------------------------------
    // 6) X Axis tick formatting
    // ------------------------------------------------
    // We only display the date on the *first tick* of each trading day;
    // otherwise, just the time (HH:MM).
    const formatDateTime = d3.timeFormat("%b %d %H:%M"); // e.g. "Aug 10 09:30"
    const formatTimeOnly = d3.timeFormat("%H:%M");       // e.g. "09:30"

    const tickFormat = (d: number, i: number, ticks: number[]) => {
      // dayOffset = floor(continuousMinutes / 390)
      const dayOffset = Math.floor(d / 390);
      const intraDay = d % 390;

      // Reconstruct a Date object for this dayOffset + minutes
      const baseDate = new Date(data[0].Date);
      baseDate.setDate(baseDate.getDate() + dayOffset);
      baseDate.setHours(9, 30, 0, 0);
      baseDate.setMinutes(baseDate.getMinutes() + intraDay);

      if (i === 0) {
        // First tick => date + time
        return formatDateTime(baseDate);
      } else {
        const prevDayOffset = Math.floor(ticks[i - 1] / 390);
        if (dayOffset !== prevDayOffset) {
          // Day changed => show date + time
          return formatDateTime(baseDate);
        } else {
          // Same day => time only
          return formatTimeOnly(baseDate);
        }
      }
    };

    // ------------------------------------------------
    // 7) Update Axes & Render functions
    // ------------------------------------------------
    function updateAxes(xScale: d3.ScaleLinear<number, number>) {
      xAxisGroup
        .call(d3.axisBottom(xScale).ticks(10).tickFormat(tickFormat as any))
        .selectAll("text")
        .attr("fill", "#333")
        .style("font-size", "12px")
        // rotate 90 degrees
        .style("text-anchor", "start")
        .attr("transform", function () {
          // Rotate around the text’s current position
          // shift the label so it doesn't overlap the axis
          return `rotate(270) translate(-70, 0)`;
        });

      xAxisGroup
        .select(".domain")
        .attr("stroke", "#666")
        .attr("stroke-width", 1.5);

      yAxisGroup
        .call(d3.axisLeft(yScale).ticks(10))
        .selectAll("text")
        .attr("fill", "#333")
        .style("font-size", "12px");
    }

    function renderChart(xScale: d3.ScaleLinear<number, number>) {
      const [minX, maxX] = xScale.domain();
      const visibleData = mappedData.filter(
        (d) => d.ContinuousTime >= minX && d.ContinuousTime <= maxX
      );

      chartContent.selectAll("*").remove(); // clear old content

      if (chartType === "line") {
        // Line chart
        const lineGenerator = d3
          .line<StockRow>()
          .x((d) => xScale(d.ContinuousTime))
          .y((d) => yScale(d.Close))
          .curve(d3.curveMonotoneX);

        chartContent
          .append("path")
          .datum(visibleData)
          .attr("fill", "none")
          .attr("stroke", "#4F46E5")
          .attr("stroke-width", 2)
          .attr("d", lineGenerator);
      } else {
        // Candlestick / Ticker
        chartContent
          .selectAll(".candle")
          .data(visibleData)
          .join("g")
          .attr("class", "candle")
          .each(function (d) {
            const group = d3.select(this);

            // Wick (high-low)
            group
              .append("line")
              .attr("x1", xScale(d.ContinuousTime))
              .attr("x2", xScale(d.ContinuousTime))
              .attr("y1", yScale(d.High))
              .attr("y2", yScale(d.Low))
              .attr("stroke", "#333");

            // Body
            group
              .append("rect")
              .attr("x", xScale(d.ContinuousTime) - 2)
              .attr("y", yScale(Math.max(d.Open, d.Close)))
              .attr("width", 4)
              .attr("height", Math.abs(yScale(d.Open) - yScale(d.Close)))
              .attr("fill", d.Open > d.Close ? "red" : "green");
          });
      }
    }

    // ------------------------------------------------
    // 8) Zoom Configuration
    // ------------------------------------------------
    // INTERVAL = smallest domain length allowed (fully zoomed in).
    // PERIOD   = the full domain (fully zoomed out).
    const INTERVAL = 15;                // e.g. 1 minute
    const PERIOD = chartSpan;          // entire dataset

    // Scale factor = (original domain size) / (new domain size).
    // domain size = chartSpan => scale = 1 (zoomed out).
    // domain size = INTERVAL => scale = chartSpan / INTERVAL (zoomed in).
    const minScale = 1;
    const maxScale = chartSpan / INTERVAL;

    const zoomBehavior = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([minScale, maxScale])
      .translateExtent([
        [margin.left, margin.top],
        [width - margin.right, height - margin.bottom],
      ])
      .on("zoom", (event) => {
        // Rescale using the transform
        const xScaleZoomed = event.transform.rescaleX(xScaleBase);

        // Clamp panning so we don't drag off the left/right
        let [minX, maxX] = xScaleZoomed.domain();

        if (minX < domainStart) {
          const shift = domainStart - minX;
          minX += shift;
          maxX += shift;
        }
        if (maxX > domainEnd) {
          const shift = maxX - domainEnd;
          minX -= shift;
          maxX -= shift;
        }

        xScaleZoomed.domain([minX, maxX] as [number, number]);

        updateAxes(xScaleZoomed);
        renderChart(xScaleZoomed);
      });

    svg.call(zoomBehavior as any);

    // ------------------------------------------------
    // 9) Initial Render
    // ------------------------------------------------
    updateAxes(xScaleBase);
    renderChart(xScaleBase);
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
