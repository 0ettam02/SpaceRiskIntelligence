"use client";

import { Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatDateShort, formatNumber } from "@/lib/formatters";

function ChartTooltip({ active, payload, label, series, valueFormatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-elevated px-3 py-2 text-xs shadow-panel">
      <p className="mb-1 font-semibold">{formatDateShort(label)}</p>
      {payload.map((entry) => {
        const meta = series.find((item) => item.key === entry.dataKey);
        return (
          <p key={entry.dataKey} className="flex items-center gap-2 text-muted">
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: entry.color }} />
            {meta?.label || entry.dataKey}: <span className="font-semibold text-ink">{valueFormatter ? valueFormatter(entry.value) : formatNumber(entry.value)}</span>
          </p>
        );
      })}
    </div>
  );
}

// Grafico temporale riutilizzabile (area singola serie oppure confronto a
// più linee sulla stessa scala): non usa mai un doppio asse Y.
export function TimeSeriesChart({ data, xKey = "date", series, height = 260, kind = "area", valueFormatter, missingDates = [] }) {
  const Chart = kind === "line" ? LineChart : AreaChart;

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <Chart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 5" vertical={false} />
          <XAxis
            dataKey={xKey}
            tickFormatter={formatDateShort}
            stroke="rgb(var(--muted))"
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgb(var(--line))" }}
            minTickGap={32}
          />
          <YAxis
            stroke="rgb(var(--muted))"
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
            tickFormatter={(value) => (valueFormatter ? valueFormatter(value) : formatNumber(value))}
          />
          <Tooltip content={<ChartTooltip series={series} valueFormatter={valueFormatter} />} />
          {series.length > 1 ? <Legend wrapperStyle={{ fontSize: 12, color: "rgb(var(--muted))" }} /> : null}
          {kind === "line"
            ? series.map((item) => (
                <Line
                  key={item.key}
                  type="monotone"
                  dataKey={item.key}
                  name={item.label}
                  stroke={item.color}
                  strokeWidth={2}
                  strokeDasharray={item.dashed ? "5 4" : undefined}
                  dot={false}
                  activeDot={{ r: 4 }}
                  isAnimationActive={false}
                />
              ))
            : series.map((item) => (
                <Area
                  key={item.key}
                  type="monotone"
                  dataKey={item.key}
                  name={item.label}
                  stroke={item.color}
                  fill={item.color}
                  fillOpacity={0.16}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
        </Chart>
      </ResponsiveContainer>
      {missingDates.length > 0 ? (
        <p className="mt-2 text-xs text-muted">
          {missingDates.length} giorni senza dati non mostrati nella finestra selezionata (segmenti non continui).
        </p>
      ) : null}
    </div>
  );
}
