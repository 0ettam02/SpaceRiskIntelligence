"use client";

import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CHART_SERIES } from "@/lib/chart-utils";
import { formatNumber, formatPercent } from "@/lib/formatters";

const COLORS = [CHART_SERIES.blue.dark, CHART_SERIES.aqua.dark];

function DistributionTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const entry = payload[0].payload;
  return (
    <div className="rounded-lg border bg-elevated px-3 py-2 text-xs shadow-panel">
      <p className="font-semibold">{entry.label}</p>
      <p className="text-muted">
        {formatNumber(entry.value)} righe · {formatPercent(entry.share)}
      </p>
    </div>
  );
}

export function ClassDistributionChart({ classes, height = 220 }) {
  const total = classes.reduce((sum, item) => sum + item.value, 0);
  const data = classes.map((item) => ({ ...item, share: total ? item.value / total : 0 }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
          <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 5" horizontal={false} />
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="label"
            width={190}
            tick={{ fill: "rgb(var(--ink))", fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<DistributionTooltip />} cursor={{ fill: "rgb(var(--elevated))" }} />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={34} isAnimationActive={false}>
            {data.map((entry, index) => (
              <Cell key={entry.id} fill={COLORS[index % COLORS.length]} />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              formatter={(value) => formatNumber(value)}
              style={{ fill: "rgb(var(--ink))", fontSize: 12, fontWeight: 600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
