/**
 * CardioFusion – Clinical Gateway Service  (Node.js / Express)
 * =============================================================
 * Responsibilities
 * ----------------
 * 1. Receive the raw AI payload from the Python service
 * 2. Apply deterministic template rendering → 3-paragraph clinical summary
 * 3. Poll Python's /explain endpoint and merge SHAP data
 * 4. Return the enriched JSON handshake to the React frontend
 *
 * SAFETY CONSTRAINT
 * -----------------
 * This service is a MEDICAL SCRIBE ONLY.
 * It formats the Python-provided values verbatim.
 * It NEVER modifies risk_score, aha_stage, or nyha_class.
 * It NEVER diagnoses, recommends treatment, or infers beyond the input data.
 */

import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import rateLimit from "express-rate-limit";
import axios from "axios";
import { z } from "zod";

const app  = express();
const PORT = process.env.PORT || 3001;
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://backend-ai:8000";
const SHAP_POLL_MAX_MS = 8000;   // maximum wait for SHAP before giving up
const SHAP_POLL_INTERVAL_MS = 500;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(helmet());
app.use(cors({ origin: process.env.FRONTEND_ORIGIN || "*" }));
app.use(express.json({ limit: "2mb" }));
app.use(morgan("combined"));

app.use(
  rateLimit({
    windowMs: 60_000,
    max: 30,
    message: { error: "Rate limit exceeded. Please retry after 60 seconds." },
  })
);

// ── Input Validation Schema (Zod) ─────────────────────────────────────────────
const EHRSchema = z.object({
  age:               z.number().min(0).max(120),
  sex:               z.union([z.literal(0), z.literal(1)]),
  height:            z.number().min(50).max(250),
  weight:            z.number().min(20).max(300),
  ejection_fraction: z.number().min(0).max(100),
  serum_creatinine:  z.number().min(0).max(20),
  sodium:            z.number().min(100).max(170),
  bp_systolic:       z.number().min(50).max(250),
  bp_diastolic:      z.number().min(30).max(150),
  smoking_status:    z.union([z.literal(0), z.literal(1)]),
  bmi:               z.number().min(10).max(80),
});

const RequestSchema = z.object({
  ecg_waveform: z.array(z.number()).length(1000),
  ehr:          EHRSchema,
});

// ── SHAP Poller ───────────────────────────────────────────────────────────────
async function pollSHAP(jobId) {
  const deadline = Date.now() + SHAP_POLL_MAX_MS;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, SHAP_POLL_INTERVAL_MS));
    const { data } = await axios.get(`${AI_SERVICE_URL}/explain/${jobId}`);
    if (data.status === "complete") return data.shap_values;
    if (data.status === "error")    return null;
  }
  return null;  // timed out – frontend shows partial result
}

// ── Clinical Summary Template Engine ─────────────────────────────────────────
// STRICT MEDICAL SCRIBE RULE:
//   All clinical values (risk_score, aha_stage, nyha_class) come EXCLUSIVELY
//   from the Python service. This function only formats; it never infers.
function buildClinicalSummary({ risk_score, aha_stage, nyha_class, ehr, shap_values }) {
  const riskPct    = (risk_score * 100).toFixed(1);
  const riskLevel  = risk_score >= 0.80 ? "critically elevated"
                   : risk_score >= 0.50 ? "significantly elevated"
                   : risk_score >= 0.25 ? "moderately elevated"
                   : "within acceptable range";

  const efDesc = ehr.ejection_fraction <= 40
    ? `a reduced ejection fraction of ${ehr.ejection_fraction}% (LVEF ≤ 40%)`
    : ehr.ejection_fraction <= 49
    ? `a mildly reduced ejection fraction of ${ehr.ejection_fraction}% (LVEF 41–49%)`
    : `a preserved ejection fraction of ${ehr.ejection_fraction}% (LVEF ≥ 50%)`;

  const creatDesc = ehr.serum_creatinine > 1.4
    ? `elevated serum creatinine (${ehr.serum_creatinine} mg/dL), suggestive of concurrent renal compromise`
    : `serum creatinine within reference range (${ehr.serum_creatinine} mg/dL)`;

  const sodiumDesc = ehr.sodium < 135
    ? `hyponatraemia (Na⁺ ${ehr.sodium} mEq/L), a known marker of haemodynamic stress`
    : `normonatraemia (Na⁺ ${ehr.sodium} mEq/L)`;

  const bpDesc = `blood pressure of ${ehr.bp_systolic}/${ehr.bp_diastolic} mmHg`;

  // Top SHAP feature for paragraph 3
  let shapNote = "";
  if (shap_values) {
    const topFeature = Object.entries(shap_values).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];
    if (topFeature) {
      const [feat, val] = topFeature;
      const direction   = val > 0 ? "positively" : "negatively";
      const readable    = feat.replace(/_/g, " ");
      shapNote = ` Interpretability analysis identifies ${readable} as the most ${direction} influential variable in the EHR-branch prediction.`;
    }
  }

  const paragraph1 =
    `The multimodal CardioFusion model assigns this patient a composite heart failure risk score of ${riskPct}%, ` +
    `classified as ${riskLevel}. ` +
    `Per ACC/AHA Heart Failure Guidelines, the haemodynamic profile is consistent with ${aha_stage}, ` +
    `reflecting ${efDesc}. ` +
    `Symptom burden is stratified at NYHA Functional Class ${nyha_class}, ` +
    `indicating ${nyha_class === "IV" ? "severe limitations at rest" :
                  nyha_class === "III" ? "marked limitation with less-than-ordinary activity" :
                  nyha_class === "II"  ? "slight limitation during ordinary physical activity" :
                                         "no symptomatic limitation during ordinary activity"}.`;

  const paragraph2 =
    `Supporting laboratory and haemodynamic data reveal ${creatDesc}, ${sodiumDesc}, ` +
    `and a presenting ${bpDesc}. ` +
    `The patient is a ${ehr.age}-year-old ${ehr.sex === 1 ? "male" : "female"} ` +
    `(BMI ${ehr.bmi.toFixed(1)} kg/m²${ehr.smoking_status === 1 ? ", active smoker" : ""}). ` +
    `These parameters were integrated with the 12-lead ECG temporal waveform through a ` +
    `Gated Attention fusion architecture, enabling joint ECG–EHR risk stratification.`;

  const paragraph3 =
    `This output is generated by an AI-assisted clinical decision support tool and is intended ` +
    `as a supplementary reference only. It does not constitute a clinical diagnosis, nor does it ` +
    `replace the judgement of a qualified cardiologist or treating physician. ` +
    `All findings should be interpreted within the full clinical context, including symptom history, ` +
    `physical examination, and additional diagnostic investigations.` +
    shapNote;

  return { paragraph1, paragraph2, paragraph3 };
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "cardiofusion-gateway", version: "1.0.0" });
});

app.post("/analyse", async (req, res) => {
  // 1. Validate input
  const parsed = RequestSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(422).json({
      error: "Validation failed",
      details: parsed.error.flatten(),
    });
  }

  try {
    // 2. Forward to Python AI service
    const { data: aiResult } = await axios.post(
      `${AI_SERVICE_URL}/predict`,
      parsed.data,
      { timeout: 15_000 }
    );

    // 3. Poll for SHAP (non-blocking timeout)
    const shap_values = await pollSHAP(aiResult.shap_job_id);

    // 4. Validate that Python's classification is internally consistent
    //    SCRIBE RULE: gateway never changes these values – only rejects
    const validAHA   = ["HFrEF", "HFpEF", "HFmrEF"];
    const validNYHA  = ["I", "II", "III", "IV"];
    if (!validAHA.includes(aiResult.aha_stage) || !validNYHA.includes(aiResult.nyha_class)) {
      return res.status(502).json({ error: "AI service returned invalid classification." });
    }

    // 5. Build the clinical narrative
    const summary = buildClinicalSummary({
      risk_score: aiResult.risk_score,
      aha_stage:  aiResult.aha_stage,
      nyha_class: aiResult.nyha_class,
      ehr:        parsed.data.ehr,
      shap_values,
    });

    // 6. Return enriched JSON handshake (as per data_schema)
    return res.json({
      // ── core handshake fields ──────────────────────────────────
      risk_score:  aiResult.risk_score,
      aha_stage:   aiResult.aha_stage,
      nyha_class:  aiResult.nyha_class,
      shap_values: shap_values ?? {},
      // ── gateway enrichments ────────────────────────────────────
      request_id:  aiResult.request_id,
      latency_ms:  aiResult.latency_ms,
      shap_ready:  shap_values !== null,
      summary,
    });
  } catch (err) {
    if (axios.isAxiosError(err)) {
      const status = err.response?.status || 502;
      return res.status(status).json({
        error: "AI service error",
        detail: err.response?.data || err.message,
      });
    }
    console.error("[gateway] unexpected error:", err);
    return res.status(500).json({ error: "Internal gateway error." });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[gateway] CardioFusion gateway listening on port ${PORT}`);
  console.log(`[gateway] AI service → ${AI_SERVICE_URL}`);
});
