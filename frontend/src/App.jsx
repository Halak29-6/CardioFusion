/**
 * CardioFusion – Main Dashboard
 * Clean & professional design: slate/white palette, card layout
 * Uses IBM Plex Sans (body) + IBM Plex Mono (data display)
 */

import React from "react";
import ECGChart    from "./components/ECGChart";
import SHAPChart   from "./components/SHAPChart";
import EHRForm     from "./components/EHRForm";
import { RiskGauge, ClassBadge } from "./components/RiskDisplay";
import { useCardioAnalysis }     from "./hooks/useCardioAnalysis";

// ── Tiny icon helpers (inline SVG – no icon lib dependency) ──────────────────
const HeartIcon = () => (
  <svg viewBox="0 0 24 24" className="w-5 h-5 fill-red-500" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
  </svg>
);

const AlertIcon = () => (
  <svg viewBox="0 0 24 24" className="w-4 h-4 fill-amber-500" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2L1 21h22L12 2zm0 3.5L20.5 19h-17L12 5.5zM11 10v4h2v-4h-2zm0 6v2h2v-2h-2z"/>
  </svg>
);

function Card({ title, badge, children, className = "" }) {
  return (
    <div className={`bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden ${className}`}>
      {title && (
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-100">
          <h2 className="text-sm font-semibold text-slate-700 tracking-wide">{title}</h2>
          {badge}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}

function StatusBadge({ text, variant = "default" }) {
  const styles = {
    default: "bg-slate-100 text-slate-600",
    success: "bg-emerald-50 text-emerald-700 border border-emerald-200",
    warning: "bg-amber-50 text-amber-700 border border-amber-200",
    error:   "bg-red-50 text-red-700 border border-red-200",
  };
  return (
    <span className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${styles[variant]}`}>
      {text}
    </span>
  );
}

function Spinner() {
  return (
    <div className="flex items-center gap-2 text-slate-400 text-sm">
      <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.37 0 0 5.37 0 12h4z"/>
      </svg>
      Analysing…
    </div>
  );
}

const NYHA_DESC = {
  I:   "No symptomatic limitation",
  II:  "Slight limitation with ordinary activity",
  III: "Marked limitation; comfortable only at rest",
  IV:  "Symptoms at rest; severe limitation",
};

const AHA_DESC = {
  HFrEF:  "Reduced Ejection Fraction (LVEF ≤ 40%)",
  HFmrEF: "Mildly Reduced EF (LVEF 41–49%)",
  HFpEF:  "Preserved Ejection Fraction (LVEF ≥ 50%)",
};

export default function App() {
  const {
    ehr, updateEhr,
    ecgData, regenerateECG,
    result, loading, shapLoading, error,
    runAnalysis,
  } = useCardioAnalysis();

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      {/* ── Header ── */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <HeartIcon />
            <div>
              <h1 className="text-lg font-bold text-slate-900 tracking-tight">CardioFusion</h1>
              <p className="text-xs text-slate-400">Multimodal Heart Failure Clinical Decision Support</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge text="ACC/AHA Guidelines" variant="default" />
            <StatusBadge text="Research Use Only" variant="warning" />
          </div>
        </div>
      </header>

      {/* ── Disclaimer Banner ── */}
      <div className="bg-amber-50 border-b border-amber-200 px-6 py-2.5">
        <div className="max-w-7xl mx-auto flex items-center gap-2 text-amber-800 text-xs">
          <AlertIcon />
          <span>
            <strong>Clinical Notice:</strong> This tool is intended for research and educational
            demonstration purposes only. It does not constitute a clinical diagnosis.
            All outputs must be reviewed by a qualified cardiologist.
          </span>
        </div>
      </div>

      {/* ── Main Layout ── */}
      <main className="max-w-7xl mx-auto px-6 py-6 grid grid-cols-12 gap-5">

        {/* LEFT COL: inputs */}
        <aside className="col-span-12 lg:col-span-4 flex flex-col gap-5">

          {/* ECG Panel */}
          <Card
            title="ECG Waveform  (Lead-II)"
            badge={
              <button
                onClick={regenerateECG}
                className="text-xs text-blue-600 hover:text-blue-800 font-medium transition"
              >
                ↺ Regenerate
              </button>
            }
          >
            <ECGChart waveform={ecgData} />
            <p className="mt-2 text-xs text-slate-400 font-mono">
              1000 samples · Z-score normalised · Lead-II proxy
            </p>
          </Card>

          {/* EHR Form */}
          <Card title="Electronic Health Record">
            <EHRForm ehr={ehr} onChange={updateEhr} />
          </Card>

          {/* Analyse Button */}
          <button
            onClick={runAnalysis}
            disabled={loading}
            className="w-full py-3.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50
              text-white font-semibold text-sm tracking-wide transition shadow-sm
              focus:outline-none focus:ring-4 focus:ring-blue-300"
          >
            {loading ? <span className="flex items-center justify-center gap-2"><Spinner /></span>
                     : "Run Multimodal Analysis"}
          </button>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </aside>

        {/* RIGHT COL: results */}
        <section className="col-span-12 lg:col-span-8 flex flex-col gap-5">

          {!result && !loading && (
            <div className="flex-1 flex items-center justify-center min-h-[400px]">
              <div className="text-center text-slate-400">
                <HeartIcon />
                <p className="mt-3 text-sm font-medium">
                  Configure patient parameters and run analysis
                </p>
                <p className="text-xs mt-1">
                  Results will appear here
                </p>
              </div>
            </div>
          )}

          {loading && (
            <div className="flex-1 flex items-center justify-center min-h-[400px]">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto" />
                <p className="mt-4 text-sm text-slate-500 font-medium">Running inference…</p>
                <p className="text-xs text-slate-400 mt-1">ECG + EHR fusion · ~200 ms</p>
              </div>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Risk + Classification row */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="sm:col-span-1">
                  <RiskGauge score={result.risk_score} />
                </div>
                <ClassBadge
                  label="AHA Stage"
                  value={result.aha_stage}
                  description={AHA_DESC[result.aha_stage]}
                />
                <ClassBadge
                  label="NYHA Class"
                  value={`Class ${result.nyha_class}`}
                  description={NYHA_DESC[result.nyha_class]}
                />
              </div>

              {/* Clinical Summary Card */}
              <Card
                title="AI-Generated Clinical Summary"
                badge={<StatusBadge text="Medical Scribe · No Diagnosis" variant="default" />}
              >
                <div className="space-y-4">
                  {[result.summary?.paragraph1, result.summary?.paragraph2, result.summary?.paragraph3]
                    .filter(Boolean)
                    .map((para, i) => (
                      <p key={i} className="text-sm text-slate-600 leading-relaxed">
                        {para}
                      </p>
                    ))}
                  {!result.summary && (
                    <p className="text-sm text-slate-400 italic">
                      Summary unavailable — gateway may be offline.
                    </p>
                  )}
                </div>
                <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-400">
                  <span className="font-mono">Request: {result.request_id?.slice(0, 8)}…</span>
                  <span>Inference: {result.latency_ms} ms</span>
                </div>
              </Card>

              {/* SHAP Interpretability */}
              <Card
                title="EHR Feature Importance  (SHAP)"
                badge={
                  shapLoading
                    ? <StatusBadge text="Calculating…" variant="warning" />
                    : <StatusBadge text="Complete" variant="success" />
                }
              >
                <SHAPChart
                  shapValues={result.shap_values}
                  loading={shapLoading}
                />
                <p className="mt-2 text-xs text-slate-400">
                  Red bars increase HF risk · Green bars decrease HF risk · Computed via SHAP KernelExplainer
                </p>
              </Card>

              {/* Raw JSON debug */}
              <details className="group">
                <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-600 select-none">
                  View raw JSON payload ▾
                </summary>
                <pre className="mt-2 bg-slate-900 text-emerald-400 rounded-xl p-4 text-xs overflow-auto max-h-64 font-mono">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </>
          )}
        </section>
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-8 mt-4 border-t border-slate-200 text-xs text-slate-400 flex flex-wrap gap-4 justify-between">
        <span>CardioFusion v1.0.0 · ACC/AHA HF Guidelines 2022 · NYHA Functional Classification</span>
        <span>TensorFlow 2.15 · FastAPI · Node.js · PTB-XL Dataset</span>
      </footer>
    </div>
  );
}
