import React, { useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  Cell, ResponsiveContainer, ReferenceLine,
} from "recharts";

const FEATURE_LABELS = {
  age:               "Age",
  sex:               "Sex",
  height:            "Height",
  weight:            "Weight",
  ejection_fraction: "Ejection Fraction",
  serum_creatinine:  "Serum Creatinine",
  sodium:            "Sodium",
  bp_systolic:       "BP Systolic",
  bp_diastolic:      "BP Diastolic",
  smoking_status:    "Smoking",
  bmi:               "BMI",
};

function SHAPTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const v = payload[0].value;
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2 shadow-md text-xs">
      <p className="font-semibold text-slate-700">{payload[0].payload.feature}</p>
      <p className={v > 0 ? "text-red-500" : "text-emerald-600"}>
        SHAP: {v > 0 ? "+" : ""}{v.toFixed(4)}
      </p>
    </div>
  );
}

export default function SHAPChart({ shapValues, loading }) {
  const data = useMemo(() => {
    if (!shapValues) return [];
    return Object.entries(shapValues)
      .map(([k, v]) => ({ feature: FEATURE_LABELS[k] || k, value: v }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }, [shapValues]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-40 gap-3">
        <div className="flex gap-1">
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="w-1.5 bg-blue-300 rounded-full animate-bounce"
              style={{ height: `${16 + i * 4}px`, animationDelay: `${i * 0.1}s` }}
            />
          ))}
        </div>
        <p className="text-xs text-slate-400 tracking-wide">Calculating interpretability…</p>
      </div>
    );
  }

  if (!data.length) {
    return (
      <p className="text-sm text-slate-400 text-center py-8">
        Run analysis to view feature importance
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20, top: 4, bottom: 4 }}>
        <XAxis
          type="number"
          tick={{ fontSize: 10, fill: "#94a3b8" }}
          tickLine={false}
          axisLine={{ stroke: "#e2e8f0" }}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fontSize: 10, fill: "#475569" }}
          tickLine={false}
          axisLine={false}
          width={110}
        />
        <Tooltip content={<SHAPTooltip />} />
        <ReferenceLine x={0} stroke="#cbd5e1" />
        <Bar dataKey="value" radius={[0, 3, 3, 0]}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.value > 0 ? "#ef4444" : "#10b981"}
              fillOpacity={0.8}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
