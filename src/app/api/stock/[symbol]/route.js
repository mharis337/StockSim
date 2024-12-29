export async function GET(request, { params }) {
  const symbol = params?.symbol;
  if (!symbol) {
    return new Response(JSON.stringify({ error: "Stock symbol is required" }), { status: 400 });
  }

  const url = new URL(request.url);
  const interval = url.searchParams.get("interval") || "1d";
  const timeframe = url.searchParams.get("timeframe") || "1mo";

  const pythonUrl = `http://localhost:8000/api/stock/${symbol}?interval=${interval}&timeframe=${timeframe}`;

  try {
    const response = await fetch(pythonUrl);
    const data = await response.json();

    if (!response.ok) {
      return new Response(JSON.stringify({ error: data.detail || "Error from Python" }), {
        status: 500,
      });
    }

    return new Response(JSON.stringify(data), { status: 200 });
  } catch (error) {
    console.error("Error calling Python backend:", error);
    return new Response(JSON.stringify({ error: "Failed to fetch from Python" }), { status: 500 });
  }
}
