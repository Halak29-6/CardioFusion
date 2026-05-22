"""
CardioFusion – Dual-Branch Fusion Network Training Script
==========================================================
Trains a multimodal Heart Failure risk model on PTB-XL data.

Architecture
------------
Branch A (ECG)  : 1D-CNN  → BatchNorm → MaxPool → Flatten
Branch B (EHR)  : MLP     → Dense-64  → Dropout → Dense-16
Fusion          : Gated Attention layer → Dense-1 Sigmoid

Preprocessing Artefacts Saved
------------------------------
  ecg_scaler.pkl   – StandardScaler fitted on ECG waveforms
  ehr_scaler.pkl   – StandardScaler fitted on EHR features
  multimodal_hf_prototype.keras – trained model

These SAME artefacts are loaded in backend-ai/main.py to guarantee
zero train/serve skew (the most common silent failure mode).
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# ── silence verbose TF logs ──────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
ARTEFACT_DIR = Path(__file__).parent / "artefacts"
ARTEFACT_DIR.mkdir(exist_ok=True)

MODEL_PATH  = ARTEFACT_DIR / "multimodal_hf_prototype.keras"
ECG_SCALER  = ARTEFACT_DIR / "ecg_scaler.pkl"
EHR_SCALER  = ARTEFACT_DIR / "ehr_scaler.pkl"

# ── constants ─────────────────────────────────────────────────────────────────
ECG_LENGTH   = 1000          # samples per waveform
N_EHR_FEATS  = 11
BATCH_SIZE   = 32
EPOCHS       = 30
EHR_FEATURES = [
    "age", "sex", "height", "weight",
    "ejection_fraction", "serum_creatinine", "sodium",
    "bp_systolic", "bp_diastolic", "smoking_status", "bmi",
]


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
#  In production: replace _synthetic_dataset() with the PTB-XL loader below.
# ═════════════════════════════════════════════════════════════════════════════

def load_ptbxl(ptbxl_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PTB-XL dataset from `ptbxl_root`.

    Returns
    -------
    ecg_data  : (N, 1000, 1)  – Lead-II waveform, raw voltage
    ehr_data  : (N, 11)       – tabular features
    labels    : (N,)          – binary HF label
    """
    import wfdb  # pip install wfdb

    df = pd.read_csv(os.path.join(ptbxl_root, "ptbxl_database.csv"), index_col="ecg_id")
    df["hf_label"] = df["scp_codes"].str.contains("NORM", na=False).astype(int)
    df["hf_label"] = (~df["scp_codes"].str.contains("NORM", na=False)).astype(int)

    ecg_list, ehr_list, label_list = [], [], []
    for ecg_id, row in df.iterrows():
        path = os.path.join(ptbxl_root, row["filename_lr"])
        signal, _ = wfdb.rdsamp(path)
        lead_ii = signal[:, 1][:ECG_LENGTH]
        if len(lead_ii) < ECG_LENGTH:
            lead_ii = np.pad(lead_ii, (0, ECG_LENGTH - len(lead_ii)))
        ecg_list.append(lead_ii)

        ehr = [
            row.get("age", 0),
            1 if row.get("sex", "M") == "M" else 0,
            row.get("height", 170),
            row.get("weight", 70),
            row.get("ejection_fraction", 55),   # synthetic EF
            row.get("serum_creatinine", 1.0),
            row.get("sodium", 140),
            row.get("bp_systolic", 120),
            row.get("bp_diastolic", 80),
            row.get("smoking_status", 0),
            row.get("weight", 70) / ((row.get("height", 170) / 100) ** 2),
        ]
        ehr_list.append(ehr)
        label_list.append(row["hf_label"])

    return (
        np.array(ecg_list, dtype=np.float32).reshape(-1, ECG_LENGTH, 1),
        np.array(ehr_list,  dtype=np.float32),
        np.array(label_list, dtype=np.float32),
    )


def _synthetic_dataset(n: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a *clinically-plausible* synthetic dataset for offline testing.
    Positive HF cases are encoded with physiologically meaningful signal so
    the model can learn a real decision boundary.
    """
    rng = np.random.default_rng(SEED)

    ecg_data  = np.zeros((n, ECG_LENGTH, 1), dtype=np.float32)
    ehr_data  = np.zeros((n, N_EHR_FEATS), dtype=np.float32)
    labels    = np.zeros(n, dtype=np.float32)

    for i in range(n):
        is_hf = rng.random() < 0.40          # 40 % prevalence

        # ── ECG waveform ─────────────────────────────────────────────────────
        t     = np.linspace(0, 4 * np.pi, ECG_LENGTH)
        hr    = rng.uniform(50, 75) if not is_hf else rng.uniform(80, 120)
        noise = rng.normal(0, 0.05, ECG_LENGTH)
        wave  = np.sin(t * hr / 60) + 0.3 * np.sin(2 * t * hr / 60) + noise
        if is_hf:
            wave += 0.4 * rng.normal(0, 1, ECG_LENGTH)   # ST-depression proxy
        ecg_data[i, :, 0] = wave.astype(np.float32)

        # ── EHR features ─────────────────────────────────────────────────────
        age   = rng.uniform(55, 85) if is_hf else rng.uniform(30, 65)
        sex   = rng.integers(0, 2)
        ht    = rng.uniform(155, 190)
        wt    = rng.uniform(60, 110)
        ef    = rng.uniform(15, 40) if is_hf else rng.uniform(50, 70)
        cr    = rng.uniform(1.2, 3.5) if is_hf else rng.uniform(0.6, 1.2)
        na    = rng.uniform(130, 138) if is_hf else rng.uniform(138, 145)
        sbp   = rng.uniform(90, 110) if is_hf else rng.uniform(110, 140)
        dbp   = rng.uniform(60, 80)
        smk   = rng.integers(0, 2)
        bmi   = wt / ((ht / 100) ** 2)

        ehr_data[i] = [age, sex, ht, wt, ef, cr, na, sbp, dbp, smk, bmi]
        labels[i]   = float(is_hf)

    return ecg_data, ehr_data, labels


# ═════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  – scalers saved here, loaded identically in FastAPI
# ═════════════════════════════════════════════════════════════════════════════

def fit_and_save_scalers(
    ecg_train: np.ndarray,
    ehr_train: np.ndarray,
) -> tuple[StandardScaler, StandardScaler]:
    ecg_flat = ecg_train.reshape(len(ecg_train), -1)

    ecg_sc = StandardScaler().fit(ecg_flat)
    ehr_sc = StandardScaler().fit(ehr_train)

    with open(ECG_SCALER, "wb") as f:
        pickle.dump(ecg_sc, f)
    with open(EHR_SCALER, "wb") as f:
        pickle.dump(ehr_sc, f)

    print(f"[scaler] ECG scaler saved → {ECG_SCALER}")
    print(f"[scaler] EHR scaler saved → {EHR_SCALER}")
    return ecg_sc, ehr_sc


def apply_scalers(
    ecg: np.ndarray,
    ehr: np.ndarray,
    ecg_sc: StandardScaler,
    ehr_sc: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(ecg)
    ecg_norm = ecg_sc.transform(ecg.reshape(n, -1)).reshape(n, ECG_LENGTH, 1)
    ehr_norm = ehr_sc.transform(ehr)
    return ecg_norm.astype(np.float32), ehr_norm.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION
# ═════════════════════════════════════════════════════════════════════════════

class GatedAttentionFusion(layers.Layer):
    """
    Learns a soft gate g ∈ (0,1) to weight ECG vs EHR contributions.
    output = g * ecg_features + (1-g) * ehr_features
    """

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gate  = layers.Dense(1, activation="sigmoid", name="attention_gate")
        self.proj  = layers.Dense(units, activation="relu",    name="fusion_proj")

    def call(self, ecg_feat: tf.Tensor, ehr_feat: tf.Tensor) -> tf.Tensor:
        combined = tf.concat([ecg_feat, ehr_feat], axis=-1)
        g        = self.gate(combined)                      # (B, 1)
        fused    = g * ecg_feat + (1.0 - g) * ehr_feat     # (B, units)
        return self.proj(fused)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def build_model(ecg_len: int = ECG_LENGTH, n_ehr: int = N_EHR_FEATS) -> keras.Model:
    # ── Branch A: ECG 1D-CNN ────────────────────────────────────────────────
    ecg_input = keras.Input(shape=(ecg_len, 1), name="ecg_input")

    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(ecg_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=4)(x)

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=4)(x)

    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=4)(x)

    ecg_feat = layers.Flatten()(x)
    ecg_feat = layers.Dense(32, activation="relu", name="ecg_embedding")(ecg_feat)

    # ── Branch B: EHR MLP ───────────────────────────────────────────────────
    ehr_input = keras.Input(shape=(n_ehr,), name="ehr_input")
    y = layers.Dense(64, activation="relu")(ehr_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(32, activation="relu")(y)
    y = layers.Dropout(0.2)(y)
    ehr_feat = layers.Dense(32, activation="relu", name="ehr_embedding")(y)

    # ── Gated Attention Fusion ───────────────────────────────────────────────
    fused  = GatedAttentionFusion(units=32, name="gated_fusion")(ecg_feat, ehr_feat)
    output = layers.Dense(1, activation="sigmoid", name="hf_risk")(fused)

    model = keras.Model(
        inputs={"ecg_input": ecg_input, "ehr_input": ehr_input},
        outputs=output,
        name="CardioFusion_DualBranch",
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train(use_ptbxl: bool = False, ptbxl_root: str | None = None) -> None:
    print("=" * 60)
    print("  CardioFusion – Training Pipeline")
    print("=" * 60)

    # 1. Load data ─────────────────────────────────────────────────────────
    if use_ptbxl and ptbxl_root:
        print("[data] Loading PTB-XL …")
        ecg, ehr, labels = load_ptbxl(ptbxl_root)
    else:
        print("[data] PTB-XL root not provided → using synthetic dataset")
        ecg, ehr, labels = _synthetic_dataset(n=3000)

    print(f"[data] ECG shape : {ecg.shape}")
    print(f"[data] EHR shape : {ehr.shape}")
    print(f"[data] Labels    : {labels.sum():.0f} positive / {len(labels)} total "
          f"({labels.mean()*100:.1f}%)")

    # 2. Split ──────────────────────────────────────────────────────────────
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=labels)
    tr, va = train_test_split(tr,  test_size=0.15, random_state=SEED, stratify=labels[tr])

    # 3. Fit scalers on TRAINING data only ─────────────────────────────────
    ecg_sc, ehr_sc = fit_and_save_scalers(ecg[tr], ehr[tr])

    ecg_tr, ehr_tr = apply_scalers(ecg[tr], ehr[tr], ecg_sc, ehr_sc)
    ecg_va, ehr_va = apply_scalers(ecg[va], ehr[va], ecg_sc, ehr_sc)
    ecg_te, ehr_te = apply_scalers(ecg[te], ehr[te], ecg_sc, ehr_sc)

    # 4. Build & compile ────────────────────────────────────────────────────
    model = build_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    # 5. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=7, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=3, mode="max", verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(ARTEFACT_DIR / "training_log.csv")),
    ]

    # 6. Train ──────────────────────────────────────────────────────────────
    history = model.fit(
        {"ecg_input": ecg_tr, "ehr_input": ehr_tr},
        labels[tr],
        validation_data=(
            {"ecg_input": ecg_va, "ehr_input": ehr_va},
            labels[va],
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate ───────────────────────────────────────────────────────────
    results = model.evaluate(
        {"ecg_input": ecg_te, "ehr_input": ehr_te},
        labels[te],
        verbose=0,
    )
    metrics = dict(zip(model.metrics_names, results))
    print("\n[eval] Test metrics:")
    for k, v in metrics.items():
        print(f"       {k:12s}: {v:.4f}")

    # 8. Save ───────────────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    print(f"\n[done] Model saved → {MODEL_PATH}")

    # 9. Save metrics manifest ──────────────────────────────────────────────
    import json
    manifest = {
        "model_path": str(MODEL_PATH),
        "ecg_scaler":  str(ECG_SCALER),
        "ehr_scaler":  str(EHR_SCALER),
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "ecg_shape":  list(ecg.shape[1:]),
        "n_ehr_feats": N_EHR_FEATS,
        "ehr_features": EHR_FEATURES,
    }
    with open(ARTEFACT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] Manifest  → {ARTEFACT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CardioFusion training")
    parser.add_argument("--ptbxl-root", default=None,
                        help="Path to PTB-XL dataset root (omit → synthetic data)")
    args = parser.parse_args()

    train(
        use_ptbxl=bool(args.ptbxl_root),
        ptbxl_root=args.ptbxl_root,
    )
