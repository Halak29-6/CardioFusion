import React, { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2 shadow-md text-xs">
      <p className="text-slate-500">Sample <span className="font-semibold text-slate-700">{label}</span></p>
      <p className="text-blue-600 font-semibold">{payload[0]?.value?.toFixed(4)} mV</p>
    </div>
  );
}

export default function ECGChart({ waveform }) {
  // Downsample 1000 pts → 250 pts for render performance
  const chartData = useMemo(() => {
    const step = 4;
    return waveform
      .filter((_, i) => i % step === 0)
      .map((v, i) => ({ sample: i * step, mv: parseFloat(v.toFixed(4)) }));
  }, [waveform]);

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={chartData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <XAxis
            dataKey="sample"
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            tickLine={false}
            axisLine={{ stroke: "#e2e8f0" }}
            label={{ value: "Samples (Lead-II, 1000pt)", position: "insideBottom", dy: 10, fontSize: 10, fill: "#94a3b8" }}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            tickLine={false}
            axisLine={false}
            domain={["auto", "auto"]}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#e2e8f0" strokeDasharray="4 2" />
          <Line
            type="monotone"
            dataKey="mv"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
            animationDuration={600}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
