"use client";

import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CHART_SERIES } from "@/lib/chart-utils";
import { formatPercent } from "@/lib/formatters";

const DEFAULT_COLOR = CHART_SERIES.blue.dark;
const RECOMMENDED_COLOR = CHART_SERIES.aqua.dark;

function BarTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const entry = payload[0].payload;
  return (
    <div className="rounded-lg border bg-elevated px-3 py-2 text-xs shadow-panel">
      <p className="font-semibold">{entry.model}</p>
      <p className="text-muted">
        Valore: <span className="font-semibold text-ink">{formatPercent(entry.value)}</span>
      </p>
      {entry.recommended ? <p className="mt-1 font-semibold text-brand-300">Modello raccomandato</p> : null}
    </div>
  );
}

// Confronto di UNA metrica alla volta fra i modelli: l'identità di ciascuna
// barra è data dall'etichetta sull'asse (posizione + testo), non dal colore,
// che qui distingue solo il modello raccomandato dagli altri.
export function MetricsBarChart({ models, metricKey, height = 300 }) {
  const data = models.map((model) => ({ model: model.model, value: model[metricKey], recommended: model.recommended }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 24, right: 12, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 5" vertical={false} />
          <XAxis
            dataKey="model"
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgb(var(--line))" }}
            interval={0}
            angle={-12}
            textAnchor="end"
            height={56}
          />
          <YAxis
            tickFormatter={(value) => formatPercent(value, { digits: 0 })}
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
            domain={[0, 1]}
          />
          <Tooltip content={<BarTooltip />} cursor={{ fill: "rgb(var(--elevated))" }} />
          <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={56} isAnimationActive={false}>
            {data.map((entry) => (
              <Cell key={entry.model} fill={entry.recommended ? RECOMMENDED_COLOR : DEFAULT_COLOR} />
            ))}
            <LabelList dataKey="value" position="top" formatter={(value) => formatPercent(value)} style={{ fill: "rgb(var(--ink))", fontSize: 11, fontWeight: 600 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
