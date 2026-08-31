"use client";

import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CHART_SERIES } from "@/lib/chart-utils";
import { formatMetric, formatNumber, formatPercent } from "@/lib/formatters";

function HistogramTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-elevated px-3 py-2 text-xs shadow-panel">
      <p className="mb-1 font-semibold">Probabilità ≈ {formatPercent(Number(label))}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} className="flex items-center gap-2 text-muted">
          <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: entry.color }} />
          {entry.name}: <span className="font-semibold text-ink">{formatNumber(entry.value)}</span>
        </p>
      ))}
    </div>
  );
}

// Istogramma dimostrativo della distribuzione delle probabilità previste per
// classe reale (vedi lib/chart-utils.generateProbabilityHistogram).
export function ProbabilityDistributionChart({ histogram, threshold, height = 280 }) {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={histogram} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 5" vertical={false} />
          <XAxis
            dataKey="bin"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => formatPercent(v, { digits: 0 })}
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgb(var(--line))" }}
          />
          <YAxis tick={{ fill: "rgb(var(--muted))", fontSize: 11 }} tickLine={false} axisLine={false} width={44} />
          <Tooltip content={<HistogramTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12, color: "rgb(var(--muted))" }} />
          <Bar dataKey="negativi" name="Classe reale: nessuna attività" fill={CHART_SERIES.blue.dark} fillOpacity={0.75} isAnimationActive={false} />
          <Bar dataKey="positivi" name="Classe reale: attività" fill={CHART_SERIES.aqua.dark} fillOpacity={0.75} isAnimationActive={false} />
          <ReferenceLine x={threshold} stroke="#e8402a" strokeWidth={2} label={{ value: `Soglia ${formatMetric(threshold, { digits: 2 })}`, position: "top", fill: "#e8402a", fontSize: 11 }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
