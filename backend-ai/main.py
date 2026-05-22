"""
CardioFusion – AI Inference Service  (FastAPI / Python 3.10)
============================================================
Endpoints
---------
POST /predict          – synchronous prediction + deterministic rules
GET  /explain/{job_id} – async SHAP result polling
GET  /health           – liveness probe
GET  /metrics          – lightweight model metadata

Design guarantees
-----------------
• Same StandardScaler artefacts used in training are loaded here
  → eliminates train/serve preprocessing skew
• Pydantic v2 validators enforce ECG array length == 1000
• SHAP runs in a BackgroundTask; main prediction returns < 200 ms
• NO PII is logged or stored
"""

from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import shap
import tensorflow as tf
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
ARTEFACT_DIR = BASE_DIR / "artefacts"
MODEL_PATH   = ARTEFACT_DIR / "multimodal_hf_prototype.keras"
ECG_SCALER   = ARTEFACT_DIR / "ecg_scaler.pkl"
EHR_SCALER   = ARTEFACT_DIR / "ehr_scaler.pkl"

# ── EHR feature order (must match training) ───────────────────────────────────
EHR_FEATURES = [
    "age", "sex", "height", "weight",
    "ejection_fraction", "serum_creatinine", "sodium",
    "bp_systolic", "bp_diastolic", "smoking_status", "bmi",
]

ECG_LENGTH = 1000

# ── in-memory SHAP job store (replace with Redis in production) ───────────────
_shap_jobs: dict[str, dict] = {}
class GatedAttentionFusion(tf.keras.layers.Layer):
    """Learns a soft gate to weight ECG vs EHR contributions."""
    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gate  = tf.keras.layers.Dense(1, activation="sigmoid", name="attention_gate")
        self.proj  = tf.keras.layers.Dense(units, activation="relu", name="fusion_proj")

    def call(self, ecg_feat: tf.Tensor, ehr_feat: tf.Tensor) -> tf.Tensor:
        combined = tf.concat([ecg_feat, ehr_feat], axis=-1)
        g        = self.gate(combined)
        fused    = g * ecg_feat + (1.0 - g) * ehr_feat
        return self.proj(fused)

    def get_config(self):
        return {**super().get_config(), "units": self.units}
app = FastAPI(
    title="CardioFusion AI Service",
    description="Multimodal HF risk inference – ECG + EHR fusion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model   = None
_ecg_sc  = None
_ehr_sc  = None
_manifest = None


@app.on_event("startup")
async def _load_artefacts() -> None:
    global _model, _ecg_sc, _ehr_sc, _manifest

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run ml-training/train.py first."
        )

    # _model = tf.keras.models.load_model(str(MODEL_PATH))
    _model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"GatedAttentionFusion": GatedAttentionFusion}
    )
    with open(ECG_SCALER, "rb") as f:
        _ecg_sc = pickle.load(f)
    with open(EHR_SCALER, "rb") as f:
        _ehr_sc = pickle.load(f)

    manifest_path = ARTEFACT_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            _manifest = json.load(f)

    print("[startup] CardioFusion model + scalers loaded successfully")


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class EHRInput(BaseModel):
    age:               float = Field(..., ge=0,   le=120,  description="Patient age in years")
    sex:               int   = Field(..., ge=0,   le=1,    description="0=Female 1=Male")
    height:            float = Field(..., ge=50,  le=250,  description="Height in cm")
    weight:            float = Field(..., ge=20,  le=300,  description="Weight in kg")
    ejection_fraction: float = Field(..., ge=0,   le=100,  description="LVEF (%)")
    serum_creatinine:  float = Field(..., ge=0,   le=20,   description="mg/dL")
    sodium:            float = Field(..., ge=100, le=170,  description="mEq/L")
    bp_systolic:       float = Field(..., ge=50,  le=250,  description="mmHg")
    bp_diastolic:      float = Field(..., ge=30,  le=150,  description="mmHg")
    smoking_status:    int   = Field(..., ge=0,   le=1,    description="0=No 1=Yes")
    bmi:               float = Field(..., ge=10,  le=80,   description="kg/m2")


class PredictRequest(BaseModel):
    ecg_waveform: list[float] = Field(
        ...,
        description="Normalised Lead-II ECG waveform; exactly 1000 samples",
    )
    ehr: EHRInput

    @field_validator("ecg_waveform")
    @classmethod
    def ecg_must_be_1000(cls, v: list[float]) -> list[float]:
        if len(v) != ECG_LENGTH:
            raise ValueError(
                f"ecg_waveform must contain exactly {ECG_LENGTH} samples; "
                f"received {len(v)}. Please resample or pad your signal."
            )
        return v


class SHAPResult(BaseModel):
    job_id:      str
    status:      Literal["pending", "complete", "error"]
    shap_values: dict[str, float] | None = None
    error:       str | None = None


class PredictResponse(BaseModel):
    request_id:  str
    risk_score:  float
    aha_stage:   Literal["HFrEF", "HFpEF", "HFmrEF"]
    nyha_class:  Literal["I", "II", "III", "IV"]
    shap_job_id: str
    latency_ms:  float


# ─── Deterministic Clinical Rules Engine ──────────────────────────────────────

def rules_engine(risk_score: float, ef: float) -> tuple[str, str]:
    """
    ACC/AHA HF Guidelines – deterministic classification.

    AHA Stage  (EF-driven):
      HFrEF  : EF <= 40
      HFmrEF : 41 <= EF <= 49
      HFpEF  : EF >= 50

    NYHA Class (risk-score proxy):
      I   : risk < 0.25
      II  : 0.25 <= risk < 0.50
      III : 0.50 <= risk < 0.80
      IV  : risk >= 0.80
    """
    if ef <= 40:
        aha_stage = "HFrEF"
    elif ef <= 49:
        aha_stage = "HFmrEF"
    else:
        aha_stage = "HFpEF"

    if risk_score >= 0.80:
        nyha_class = "IV"
    elif risk_score >= 0.50:
        nyha_class = "III"
    elif risk_score >= 0.25:
        nyha_class = "II"
    else:
        nyha_class = "I"

    return aha_stage, nyha_class


# ─── SHAP Background Computation ─────────────────────────────────────────────

def _compute_shap(job_id: str, ehr_norm: np.ndarray) -> None:
    try:
        ehr_branch = tf.keras.Model(
            inputs=_model.input["ehr_input"],
            outputs=_model.get_layer("ehr_embedding").output,
        )

        def ehr_predict(x: np.ndarray) -> np.ndarray:
            return ehr_branch.predict(x, verbose=0)

        background = np.zeros((10, ehr_norm.shape[1]), dtype=np.float32)
        explainer  = shap.KernelExplainer(ehr_predict, background)
        shap_vals  = explainer.shap_values(ehr_norm, nsamples=50)

        sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
        if sv.ndim == 3:
            sv = np.linalg.norm(sv, axis=-1)

        feature_importance = {
            feat: float(sv[0, i]) for i, feat in enumerate(EHR_FEATURES)
        }

        _shap_jobs[job_id] = {"status": "complete", "shap_values": feature_importance}

    except Exception as exc:
        _shap_jobs[job_id] = {"status": "error", "error": str(exc)}


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {
        "status":         "ok",
        "model_loaded":   _model is not None,
        "scalers_loaded": _ecg_sc is not None and _ehr_sc is not None,
    }


@app.get("/metrics", tags=["ops"])
async def metrics() -> dict:
    if _manifest is None:
        return {"message": "manifest not found"}
    return {
        "model_version":    "1.0.0",
        "test_metrics":     _manifest.get("test_metrics", {}),
        "ehr_features":     EHR_FEATURES,
        "ecg_input_length": ECG_LENGTH,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(body: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    t0 = time.perf_counter()

    if _model is None or _ecg_sc is None or _ehr_sc is None:
        raise HTTPException(503, "Model artefacts not loaded.")

    # 1. Preprocess ECG – Z-score via training scaler
    ecg_raw  = np.array(body.ecg_waveform, dtype=np.float32).reshape(1, -1)
    ecg_norm = _ecg_sc.transform(ecg_raw).reshape(1, ECG_LENGTH, 1)

    # 2. Preprocess EHR – same scaler as training
    ehr_raw  = np.array([getattr(body.ehr, f) for f in EHR_FEATURES], dtype=np.float32).reshape(1, -1)
    ehr_norm = _ehr_sc.transform(ehr_raw).astype(np.float32)

    # 3. Inference
    preds      = _model.predict({"ecg_input": ecg_norm, "ehr_input": ehr_norm}, verbose=0)
    risk_score = float(preds[0, 0])

    # 4. Deterministic rules override
    aha_stage, nyha_class = rules_engine(risk_score, body.ehr.ejection_fraction)

    # 5. Queue SHAP (non-blocking – poll /explain/{job_id})
    job_id = str(uuid.uuid4())
    _shap_jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(_compute_shap, job_id, ehr_norm)

    return PredictResponse(
        request_id=  str(uuid.uuid4()),
        risk_score=  round(risk_score, 4),
        aha_stage=   aha_stage,
        nyha_class=  nyha_class,
        shap_job_id= job_id,
        latency_ms=  round((time.perf_counter() - t0) * 1000, 2),
    )


@app.get("/explain/{job_id}", response_model=SHAPResult, tags=["inference"])
async def explain(job_id: str) -> SHAPResult:
    job = _shap_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"SHAP job '{job_id}' not found.")
    return SHAPResult(
        job_id=      job_id,
        status=      job["status"],
        shap_values= job.get("shap_values"),
        error=       job.get("error"),
    )
