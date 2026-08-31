"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CHART_SERIES } from "@/lib/chart-utils";
import { formatMetric, formatPercent } from "@/lib/formatters";

function PrTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="rounded-lg border bg-elevated px-3 py-2 text-xs shadow-panel">
      <p>Recall: {formatPercent(point.recall)}</p>
      <p>Precision: {formatPercent(point.precision)}</p>
    </div>
  );
}

// Curva Precision-Recall dimostrativa (vedi lib/chart-utils.generatePrCurve).
export function PrCurveChart({ data, prAuc, height = 260 }) {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 5" />
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => formatPercent(v, { digits: 0 })}
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgb(var(--line))" }}
            label={{ value: "Recall", position: "insideBottom", offset: -2, fill: "rgb(var(--muted))", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => formatPercent(v, { digits: 0 })}
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
            label={{ value: "Precision", angle: -90, position: "insideLeft", fill: "rgb(var(--muted))", fontSize: 11 }}
          />
          <Tooltip content={<PrTooltip />} />
          <Line
            type="monotone"
            dataKey="precision"
            name={`Modello (PR-AUC ${formatMetric(prAuc)})`}
            stroke={CHART_SERIES.magenta.dark}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
