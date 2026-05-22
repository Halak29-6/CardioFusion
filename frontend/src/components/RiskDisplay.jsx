import React from "react";

const RISK_CONFIG = {
  low:      { label: "Low Risk",      bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-700", bar: "bg-emerald-500", ring: "ring-emerald-200" },
  moderate: { label: "Moderate Risk", bg: "bg-amber-50",   border: "border-amber-200",   text: "text-amber-700",   bar: "bg-amber-500",   ring: "ring-amber-200"   },
  high:     { label: "High Risk",     bg: "bg-orange-50",  border: "border-orange-200",  text: "text-orange-700",  bar: "bg-orange-500",  ring: "ring-orange-200"  },
  critical: { label: "Critical Risk", bg: "bg-red-50",     border: "border-red-200",     text: "text-red-700",     bar: "bg-red-500",     ring: "ring-red-200"     },
};

function getRiskLevel(score) {
  if (score >= 0.80) return "critical";
  if (score >= 0.50) return "high";
  if (score >= 0.25) return "moderate";
  return "low";
}

export function RiskGauge({ score }) {
  const level  = getRiskLevel(score);
  const config = RISK_CONFIG[level];
  const pct    = Math.round(score * 100);

  return (
    <div className={`rounded-xl border ${config.border} ${config.bg} p-5`}>
      <p className="text-xs font-medium text-slate-500 uppercase tracking-widest mb-1">
        Composite HF Risk
      </p>
      <div className="flex items-end gap-2 mb-3">
        <span className={`text-5xl font-bold tabular-nums ${config.text}`}>{pct}</span>
        <span className={`text-xl font-medium mb-1 ${config.text}`}>%</span>
        <span className={`ml-auto text-sm font-semibold px-2.5 py-1 rounded-full ${config.bg} ${config.text} border ${config.border} ring-2 ${config.ring}`}>
          {config.label}
        </span>
      </div>
      <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full ${config.bar} rounded-full transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function ClassBadge({ label, value, description }) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-col gap-1">
      <p className="text-xs text-slate-400 uppercase tracking-widest font-medium">{label}</p>
      <p className="text-2xl font-bold text-slate-800">{value}</p>
      {description && <p className="text-xs text-slate-500">{description}</p>}
    </div>
  );
}
