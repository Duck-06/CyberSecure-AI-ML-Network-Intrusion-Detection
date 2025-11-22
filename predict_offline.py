# predict_offline.py — 100% offline ML + Blockchain logging (NO TRAIN CSV REQUIRED)

import pandas as pd
import joblib
import time
from pathlib import Path
from blockchain import Blockchain

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")
MODEL_PATH = DATA_DIR / "xgboost_intrusion_model_high_recall.pkl"
SCALER_PATH = DATA_DIR / "scaler.pkl"

INPUT_FILE = "inputs.csv"     # your input file with flow data (must contain feature columns)

# Full feature list used during training
FEATURE_COLS = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Length of Fwd Packets",
    "Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s",
    "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","PSH Flag Count","ACK Flag Count","Average Packet Size","Subflow Fwd Bytes",
    "Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Max","Active Min","Idle Mean","Idle Max","Idle Min"
]

# ---------------- LOAD MODEL + SCALER ----------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------- LOAD INPUT FILE ----------------
if INPUT_FILE.lower().endswith(".csv"):
    df = pd.read_csv(INPUT_FILE)
elif INPUT_FILE.lower().endswith(".xlsx"):
    df = pd.read_excel(INPUT_FILE)
else:
    raise ValueError("Supported formats: CSV or XLSX only")

# ---------------- VALIDATE COLUMNS ----------------
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Your input file is missing required feature columns: {missing}")

# Keep only required model features
df = df[FEATURE_COLS].copy()

# ---------------- SCALE INPUT ----------------
scaled = scaler.transform(df)

# ---------------- PREDICTION ----------------
probs = model.predict_proba(scaled)[:, 1]   # intrusion probability
labels = (probs >= 0.5).astype(int)         # default threshold = 0.5

# ---------------- BLOCKCHAIN INIT ----------------
bc = Blockchain()
summary = []

# ---------------- APPEND ONE BLOCK PER FLOW ----------------
for i in range(len(df)):
    prob = float(probs[i])
    pred = int(labels[i])

    confidence_not_intr = (1 - prob) * 100        # % confidence benign
    confidence_intr = prob * 100                  # % confidence intrusion
    final_confidence = max(confidence_not_intr, confidence_intr)

    event = {
        "type": "ml_prediction",
        "prediction": pred,
        "intrusion_probability": prob,
        "confidence_percentage": float(final_confidence),
        "timestamp": time.time(),
        "model_version": "xgboost_high_recall_v1"
    }

    block = bc.create_block(event)
    summary.append(block)

# ---------------- OUTPUT ----------------
print("\n=== Appended Blockchain Blocks ===")
summary_df = pd.DataFrame(summary)
print(summary_df[["index", "hash", "prev_hash"]])

# Save CSV summary
summary_df.to_csv("prediction_block_summary.csv", index=False)
print("Saved: prediction_block_summary.csv")
