import React from "react";

const FIELDS = [
  { key: "age",               label: "Age",               unit: "yrs",   min: 0,   max: 120, step: 1    },
  { key: "sex",               label: "Sex",               unit: "0=F 1=M", min: 0, max: 1,   step: 1    },
  { key: "height",            label: "Height",            unit: "cm",    min: 50,  max: 250, step: 0.5  },
  { key: "weight",            label: "Weight",            unit: "kg",    min: 20,  max: 300, step: 0.5  },
  { key: "ejection_fraction", label: "Ejection Fraction", unit: "LVEF%", min: 0,   max: 100, step: 1    },
  { key: "serum_creatinine",  label: "Serum Creatinine",  unit: "mg/dL", min: 0,   max: 20,  step: 0.1  },
  { key: "sodium",            label: "Sodium",            unit: "mEq/L", min: 100, max: 170, step: 1    },
  { key: "bp_systolic",       label: "BP Systolic",       unit: "mmHg",  min: 50,  max: 250, step: 1    },
  { key: "bp_diastolic",      label: "BP Diastolic",      unit: "mmHg",  min: 30,  max: 150, step: 1    },
  { key: "smoking_status",    label: "Smoking",           unit: "0=No 1=Yes", min: 0, max: 1, step: 1   },
  { key: "bmi",               label: "BMI",               unit: "kg/m²", min: 10,  max: 80,  step: 0.1  },
];

// EF-based colour hint
function efHint(ef) {
  if (ef <= 40)  return "text-red-500";
  if (ef <= 49)  return "text-amber-500";
  return "text-emerald-600";
}

export default function EHRForm({ ehr, onChange }) {
  return (
    <div className="grid grid-cols-1 gap-3">
      {FIELDS.map(({ key, label, unit, min, max, step }) => (
        <div key={key} className="flex items-center gap-3">
          <label className="text-xs text-slate-500 w-36 shrink-0 font-medium">{label}</label>
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={ehr[key]}
            onChange={(e) => onChange(key, e.target.value)}
            className={`w-full border rounded-lg px-2.5 py-1.5 text-sm tabular-nums
              focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent
              border-slate-200 bg-white text-slate-800 transition
              ${key === "ejection_fraction" ? efHint(ehr.ejection_fraction) : "text-slate-800"}
            `}
          />
          <span className="text-xs text-slate-400 w-14 text-right shrink-0">{unit}</span>
        </div>
      ))}
    </div>
  );
}
