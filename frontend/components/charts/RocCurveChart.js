"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CHART_SERIES } from "@/lib/chart-utils";
import { formatMetric, formatPercent } from "@/lib/formatters";

function RocTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="rounded-lg border bg-elevated px-3 py-2 text-xs shadow-panel">
      <p>Falsi positivi: {formatPercent(point.fpr)}</p>
      <p>Veri positivi: {formatPercent(point.tpr)}</p>
    </div>
  );
}

// Curva ROC dimostrativa (vedi lib/chart-utils.generateRocCurve): la forma
// riflette l'AUC reale del modello ma i punti non sono osservazioni dirette.
export function RocCurveChart({ data, auc, height = 260 }) {
  const diagonal = data.map((point) => ({ ...point, chance: point.fpr }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <LineChart data={diagonal} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 5" />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => formatPercent(v, { digits: 0 })}
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgb(var(--line))" }}
            label={{ value: "Tasso di falsi positivi", position: "insideBottom", offset: -2, fill: "rgb(var(--muted))", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => formatPercent(v, { digits: 0 })}
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
            label={{ value: "Tasso di veri positivi", angle: -90, position: "insideLeft", fill: "rgb(var(--muted))", fontSize: 11 }}
          />
          <Tooltip content={<RocTooltip />} />
          <Line type="monotone" dataKey="chance" name="Classificatore casuale" stroke="rgb(var(--muted))" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="tpr" name={`Modello (AUC ${formatMetric(auc)})`} stroke={CHART_SERIES.blue.dark} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
