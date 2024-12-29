"use client";

import React, { useEffect, useRef } from "react";
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
  interval: string;
}

const StockChart: React.FC<StockChartProps> = ({ data, interval }) => {
  const chartRef = useRef<SVGSVGElement | null>(null);

  const intradayIntervals = ["1m", "15m", "30m", "60m", "90m", "1h"];
  const isIntraday = intradayIntervals.includes(interval);

  useEffect(() => {
    if (!data || !Array.isArray(data) || data.length === 0 || !chartRef.current) return;

    const svg = d3.select(chartRef.current);
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };

    svg.selectAll("*").remove(); // Clear previous content

    // Initial xScale
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(data, (d) => new Date(d.Date)) as [Date, Date])
      .range([margin.left, width - margin.right]);

    // Save the original domain
    const originalDomain = xScale.domain();

    const yScale = d3
      .scaleLinear()
      .domain([
        d3.min(data, (d) => d.Low) || 0,
        d3.max(data, (d) => d.High) || 0,
      ])
      .nice()
      .range([height - margin.bottom, margin.top]);

    svg.append("defs")
      .append("clipPath")
      .attr("id", "clip")
      .append("rect")
      .attr("x", margin.left)
      .attr("y", margin.top)
      .attr("width", width - margin.left - margin.right)
      .attr("height", height - margin.top - margin.bottom);

    const chartContent = svg
      .append("g")
      .attr("clip-path", "url(#clip)");

    const xAxisGroup = svg
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height - margin.bottom})`);

    const yAxisGroup = svg
      .append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`);

    const updateAxes = (newXScale: any) => {
      xAxisGroup
        .call(
          d3.axisBottom(newXScale).tickFormat((d) =>
            isIntraday
              ? d3.timeFormat("%H:%M")(d as Date)
              : d3.timeFormat("%b %d")(d as Date)
          )
        )
        .selectAll("text")
        .attr("fill", "#333")
        .style("font-size", "12px");

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

    // Initial render of axes
    updateAxes(xScale);

    chartContent
      .selectAll(".candle")
      .data(data)
      .join("g")
      .attr("class", "candle")
      .each(function (d) {
        const group = d3.select(this);
        group.append("line")
          .attr("x1", xScale(new Date(d.Date)))
          .attr("x2", xScale(new Date(d.Date)))
          .attr("y1", yScale(d.High))
          .attr("y2", yScale(d.Low))
          .attr("stroke", "#333");

        group.append("rect")
          .attr("x", xScale(new Date(d.Date)) - 2)
          .attr("y", yScale(Math.max(d.Open, d.Close)))
          .attr("width", 4)
          .attr("height", Math.abs(yScale(d.Open) - yScale(d.Close)))
          .attr("fill", d.Open > d.Close ? "red" : "green");
      });

    const zoom = d3
      .zoom()
      .scaleExtent([1, 5])
      .translateExtent([
        [margin.left, margin.top],
        [width - margin.right, height - margin.bottom],
      ])
      .on("zoom", (event) => {
        const { transform } = event;

        // Update scales with zoom
        const zoomedXScale = transform.rescaleX(xScale);

        // Update axes and chart content
        updateAxes(zoomedXScale);

        chartContent
          .selectAll(".candle")
          .attr("transform", (d) => `translate(${zoomedXScale(new Date(d.Date)) - xScale(new Date(d.Date))}, 0)`);

        // Reset x-axis when zooming out completely
        if (transform.k === 1) {
          updateAxes(xScale.domain(originalDomain));
        }
      });

    svg.call(zoom);
  }, [data, interval]);

  return <svg ref={chartRef} width="800" height="400"></svg>;
};

export default StockChart;
